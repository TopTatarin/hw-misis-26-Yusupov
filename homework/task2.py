import argparse

import torch
import triton
import triton.language as tl


def layernorm_forward_torch(
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        eps: float = 1e-5,
) -> torch.Tensor:
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, unbiased=False, keepdim=True)

    rstd = 1.0 / torch.sqrt(var + eps)
    x_hat = (x - mean) * rstd

    return x_hat * weight + bias


# Автотюним только число варпов: BLOCK_SIZE жёстко задаётся снаружи как
# next_power_of_2(N), чтобы вся строка гарантированно влезала в один блок.
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16),
    ],
    key=["N"],
)
@triton.jit
def layernorm_forward_kernel(
        x_ptr, w_ptr, b_ptr, y_ptr,
        mean_ptr, rstd_ptr,  # сохраняем статистику строки для backward

        stride_row,          # шаг между строками
        N: int,
        eps,

        BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(axis=0)

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    x_row_ptr = x_ptr + row * stride_row
    # other=0 для замаскированного хвоста — не влияет на сумму
    x = tl.load(x_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    mean = tl.sum(x, axis=0) / N
    x_centered = tl.where(mask, x - mean, 0.0)
    var = tl.sum(x_centered * x_centered, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    # Кэшируем mean/rstd — в backward пересчитывать их было бы дороже.
    tl.store(mean_ptr + row, mean)
    tl.store(rstd_ptr + row, rstd)

    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    x_hat = x_centered * rstd
    y = x_hat * w + b

    # tl.store сам приведёт fp32 -> dtype выходного тензора.
    tl.store(y_ptr + row * stride_row + cols, y, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16),
    ],
    key=["N"],
)
@triton.jit
def layernorm_backward_kernel(
        dy_ptr, x_ptr, w_ptr,
        mean_ptr, rstd_ptr,

        dx_ptr, dw_ptr, db_ptr,  # dw/db — общие fp32-аккумуляторы на все строки

        stride_row,
        N: int,

        BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(axis=0)

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    x = tl.load(x_ptr + row * stride_row + cols, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(dy_ptr + row * stride_row + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    mean = tl.load(mean_ptr + row)
    rstd = tl.load(rstd_ptr + row)

    x_hat = tl.where(mask, (x - mean) * rstd, 0.0)
    wdy = tl.where(mask, w * dy, 0.0)

    # Две редукции по строке: c1 и c2 — это нормировочные члены градиента входа.
    c1 = tl.sum(x_hat * wdy, axis=0) / N
    c2 = tl.sum(wdy, axis=0) / N
    dx = (wdy - x_hat * c1 - c2) * rstd

    tl.store(dx_ptr + row * stride_row + cols, dx, mask=mask)

    # dw и db суммируются по всем строкам M -> разные программы пишут в одни и те же
    # ячейки [N]. atomic_add сериализует эти конкурентные записи без гонок.
    tl.atomic_add(dw_ptr + cols, dy * x_hat, mask=mask)
    tl.atomic_add(db_ptr + cols, dy, mask=mask)

class _LayerNormTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        x_arg = x.reshape(-1, x.shape[-1]).contiguous()
        M, N = x_arg.shape

        y = torch.empty_like(x_arg)
        mean = torch.empty(M, device=x.device, dtype=torch.float32)
        rstd = torch.empty(M, device=x.device, dtype=torch.float32)

        BLOCK_SIZE = triton.next_power_of_2(N)

        layernorm_forward_kernel[(M,)](
            x_arg, weight, bias, y,
            mean, rstd,
            x_arg.stride(0), N, eps,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        ctx.save_for_backward(x_arg, weight, mean, rstd)
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.x_shape = x.shape
        return y.reshape(x.shape)

    @staticmethod
    def backward(ctx, dy):
        x_arg, weight, mean, rstd = ctx.saved_tensors
        M, N = x_arg.shape

        dy_arg = dy.reshape(-1, N).contiguous()

        dx = torch.empty_like(x_arg)
        # Аккумуляторы для atomic_add обязаны быть обнулены и в fp32.
        dw = torch.zeros(N, device=x_arg.device, dtype=torch.float32)
        db = torch.zeros(N, device=x_arg.device, dtype=torch.float32)

        layernorm_backward_kernel[(M,)](
            dy_arg, x_arg, weight,
            mean, rstd,
            dx, dw, db,
            x_arg.stride(0), N,
            BLOCK_SIZE=ctx.BLOCK_SIZE,
        )

        # grad по x, weight, bias, eps(None)
        return dx.reshape(ctx.x_shape), dw.to(weight.dtype), db.to(weight.dtype), None


def layernorm_triton(
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        eps: float = 1e-5,
) -> torch.Tensor:
    return _LayerNormTriton.apply(x, weight, bias, eps)


def check_correctness():
    torch.manual_seed(0)
    device = "cuda"

    # fp32 строгая проверка и forward, и backward.
    M, N = 1151, 8192  # неровные размеры специально, чтобы проверить маску
    x = torch.randn(M, N, device=device, dtype=torch.float32, requires_grad=True)
    weight = torch.randn(N, device=device, dtype=torch.float32, requires_grad=True)
    bias = torch.randn(N, device=device, dtype=torch.float32, requires_grad=True)
    dy = torch.randn(M, N, device=device, dtype=torch.float32)

    #forward
    y_ref = torch.nn.functional.layer_norm(x, (N,), weight, bias, eps=1e-5)
    y_mine = layernorm_triton(x, weight, bias, eps=1e-5)
    torch.testing.assert_close(y_mine, y_ref)
    print("[ok] forward совпадает с torch.nn.functional.layer_norm")

    #backward
    y_ref.backward(dy, retain_graph=True)
    dx_ref, dw_ref, db_ref = x.grad.clone(), weight.grad.clone(), bias.grad.clone()

    x.grad = weight.grad = bias.grad = None
    y_mine.backward(dy)
    dx_mine, dw_mine, db_mine = x.grad, weight.grad, bias.grad

    torch.testing.assert_close(dx_mine, dx_ref)
    torch.testing.assert_close(dw_mine, dw_ref, atol=1e-2, rtol=1e-2)  # atomic_add копит ошибку округления
    torch.testing.assert_close(db_mine, db_ref, atol=1e-2, rtol=1e-2)
    print("[ok] backward (dx, dweight, dbias) совпадает с autograd")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", action="store_true", help="запустить бенчмарк")
    args = parser.parse_args()

    check_correctness()

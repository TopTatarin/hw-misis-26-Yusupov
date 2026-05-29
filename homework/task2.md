# ДЗ №2

## Задача

На занятии мы разобрали, как писать кернелы на Triton для простых поэлементных операций, таких как `SiLU-Mul`.

Ваша следующая задача — написать forward pass и backward pass (две разных функции) для операции **Layer Normalization** (`torch.nn.LayerNorm`).

1. Реализуйте forward pass.
2. Реализуйте backward pass.
3. Проверьте решение на корректность через `torch.testing.assert_close`.
4. Добавьте autotune.
5. Сравните скорость вашего решения со скоростью эталонного решения на PyTorch.

Эталонное решение на PyTorch:

```python
def layernorm_forward_torch(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5):
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, unbiased=False, keepdim=True)
    
    rstd = 1.0 / torch.sqrt(var + eps)
    x_hat = (x - mean) * rstd
    
    # 4. Применяем scale (weight) и shift (bias)
    output = x_hat * weight + bias
    
    return output
```

*Подсказка:* Пусть дана матрица `[M; N]`, где M - количество элементов, а N - hidden size; в гриде вы будете запускаться по оси `M`, имея доступ ко всем `N` в рамках выбранных (или выбранного) `M`, - это знание упростит подсчёт статистики.

*Подсказка:* Для аккумуляции градиентов в backward pass вам может понадобиться функция `tl.atomic_add`: [документация](https://triton-lang.org/main/python-api/generated/triton.language.atomic_add.html).


## Сдача

Файл с исходным кодом в виде ссылки на путь в GitHub-репозитории нужно отправить мне в личку в телеграм. 

Если вы сделали бенчмарк, приложите к репозиторию/сообщению скриншот графика или просто цифры из текстового репорта пропускной способности.

### Баллы
В зависимости от того, какую часть задания вы сделаете:

* **3 балла** - только forward pass, с проверкой корректности;
* **4 балла** - forward pass и backward pass, с проверкой корректности;
* **5 баллов** - forward pass и backward pass, с проверкой корректности, автотюном и бенчмарком.

### Дедлайн

Сдавать и корректировать решение можно до **29 мая**.

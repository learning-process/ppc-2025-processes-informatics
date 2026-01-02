
# Примеры функций для global_search_strongin

В этой задаче вход содержит **функцию** `objective(x)` (как `std::function<double(double)>`).
Сам объект-функцию в общем виде в файл «данных» не записать без отдельного языка выражений/парсера.

Поэтому для `data` сделаны два вида «настоящих данных»:
1) параметры кейсов (как числа);
2) дискретизации функций (таблицы `x, f(x)`), которые можно открыть/проверить как обычные данные.

Ниже — набор простых функций, которые удобно использовать в ручной проверке и в тестах.

## 1) Квадратичная (гладкая, один минимум)

Отрезок: `[-5, 5]`

```cpp
input.left = -5.0;
input.right = 5.0;
input.epsilon = 1e-3;
input.reliability = 2.0;
input.max_iterations = 250;
input.objective = [](double x) { return (x - 2.0) * (x - 2.0); };
```

Ожидаемо: минимум около `x = 2`.

## 2) Sin + линейный тренд (гладкая, много локальных экстремумов)

Отрезок: `[-10, 10]`

```cpp
input.left = -10.0;
input.right = 10.0;
input.epsilon = 1e-3;
input.reliability = 2.0;
input.max_iterations = 300;
input.objective = [](double x) { return std::sin(x) + 0.1 * x; };
```

## 3) |x| + cos(10x) (негладкая в 0, много локальных минимумов)

Отрезок: `[-3, 3]`

```cpp
input.left = -3.0;
input.right = 3.0;
input.epsilon = 1e-3;
input.reliability = 2.0;
input.max_iterations = 300;
input.objective = [](double x) { return std::fabs(x) + 0.2 * std::cos(10.0 * x); };
```

## Файлы в `data`

- `cases_sample.csv` — параметры кейсов (границы, `epsilon`, `reliability`, `max_iterations`, описание функции).
- `reference_minima.csv` — оценка минимума и аргминимума по плотной сетке (20000 отсчётов), как в тестах.
- `quadratic_samples.csv`, `sin_linear_samples.csv`, `abs_cos_samples.csv` — таблицы `x, f(x)` (201 точка на отрезке).

Если понадобится запуск «по файлу данных», то тогда да — придётся договориться о формате описания функции и написать парсер.


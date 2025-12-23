# Линейная фильтрация изображений (блочное разбиение). Ядро Гаусса 3x3.
**Студент:** Зенин Антон Алексеевич.
**Группа:** 3823Б1ФИ1
**Технология:** SEQ | MPI
**Вариант:** 28 
---

## 1. Введение
  


---

## 2. Постановка задачи



**Входные данные**  

  

**Выходные данные**



---

## 3. Описание алгоритма SEQ (Sequential)

 


## 4. Описание алгоритма MPI  

 


## 5. Особенности реализации  

## Структура кода 

common/include/common.hpp — InType, OutType, формат входных и выходных данных  
seq/src/ops_seq.cpp — SEQ-реализация     
mpi/src/ops_mpi.cpp — MPI-реализация   
tests/functional/main.cpp — functional tests   
tests/performance/main.cpp — performance tests  

## Классы

- `ZeninATopologyStarMPI : BaseTask`   
- `ZeninATopologyStarSEQ : BaseTask`   


## 6. Окружение 


### Hardware
- CPU: Intel(R) Core(TM) i5-10400F CPU @ 2.90GHz    
- Cores: 6    
- RAM: 32 GB    
- OS: Windows 10 Pro  

### Toolchain
- Compiler: `C++20`     
- MPI: OpenMPI    
- Build type: Release  

## Данные  


---

## Результаты и выводы  

### 7.1 Корректность  


### 7.2 Производительность  

#### Результаты performance-тестов  

 


---  

## Анализ результатов  



---

## 8. Выводы  


---

## 9. Источники
- cppreference.com - https://en.cppreference.com  
- Документация по OpenMPI - https://www.open-mpi.org/doc  
- Лекции по параллельному программированию ННГУ им. Лобачевского  
- Практические занятия по параллельному программированию ННГУ им. Лобачевского  

---  

## 10. Приложение. Код MPI реализации  
```cpp

```

  
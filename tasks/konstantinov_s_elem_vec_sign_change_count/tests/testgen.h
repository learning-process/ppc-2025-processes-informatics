#pragma once
#include <cstring>  // memcpy
#include <vector>
#include <iostream>

// СТАБИЛЬНЫЙ генератор тестовых данных
// Нужно подать массив с известным колвом смен знаков и одинаковыми знаками первого и последнего элемента
inline int generateTestData(const int *examplearr, size_t arrsz, int sign_change_count, std::vector<int> &v) {
  size_t m = v.size();
  //std::cout<<"Generating "<<m<<"\n";
  if (arrsz == 0 || m == 0) {
    return 0;
  }

  if (m < arrsz)  // для особо маленьких размеров
  {
    int res = 0;
    for (int i = 0; i < m - 1; i++) {
      v[i] = examplearr[i];
      res += (examplearr[i] > 0) != (examplearr[i + 1] > 0);
    }
    v[m - 1] = examplearr[m - 1];
    return res;
  }

  size_t fullBlocks = m / arrsz;
  size_t tail = m % arrsz;

  int *dst = v.data();

  // Копируем первую копию массива
  memcpy(dst, examplearr, arrsz * sizeof(int));
  size_t filled = 1;  // число заполненных блоков

  // Удвоительное копирование блоками
  // пока удвоение не превысит количество нужных полных блоков
  while (filled * 2 <= fullBlocks) {
    memcpy(dst + filled * arrsz, dst, filled * arrsz * sizeof(int));
    filled *= 2;
  }

  // Дозаполняем оставшиеся полные блоки
  while (filled < fullBlocks) {
    memcpy(dst + filled * arrsz, dst, arrsz * sizeof(int));
    filled++;
  }

  // Хвост = последний элемент чтобы не считать смены знаков там
  if (tail > 0) {
    int last = examplearr[arrsz - 1];
    for (size_t i = 0; i < tail; ++i) {
      dst[fullBlocks * arrsz + i] = last;
    }
  }

//   for(int i=0;i<m;i++)
//        std::cout<<v[i]<<" ";
//      std::cout<<"\n\n";

  return sign_change_count * fullBlocks;
}

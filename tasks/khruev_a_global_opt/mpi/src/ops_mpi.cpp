#include "khruev_a_global_opt/mpi/include/ops_mpi.hpp"
#include <algorithm>
#include <cmath>
#include <vector>

namespace khruev_a_global_opt {

struct IntervalInfo {
    double R;
    int index; // индекс в массиве trials_ (указывает на правый конец интервала)
    
    bool operator>(const IntervalInfo& other) const {
        return R > other.R; // Для сортировки по убыванию
    }
};

KhruevAGlobalOptMPI::KhruevAGlobalOptMPI(const InType& in) {
    SetTypeOfTask(GetStaticTypeOfTask());
    GetInput() = in;
}

bool KhruevAGlobalOptMPI::ValidationImpl() {
    return GetInput().max_iter > 0 && GetInput().epsilon > 0 && GetInput().r > 1.0;
}

bool KhruevAGlobalOptMPI::PreProcessingImpl() {
    trials_.clear();
    return true;
}

double KhruevAGlobalOptMPI::CalculateFunction(double t) {
    double u, v;
    d2xy(t, u, v);
    double real_x = GetInput().ax + u * (GetInput().bx - GetInput().ax);
    double real_y = GetInput().ay + v * (GetInput().by - GetInput().ay);
    return target_function(GetInput().func_id, real_x, real_y);
}

void KhruevAGlobalOptMPI::AddTrialUnsorted(double t, double z) {
    Trial tr;
    tr.x = t;
    tr.z = z;
    trials_.push_back(tr);
}

bool KhruevAGlobalOptMPI::RunImpl() {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    AddTrialUnsorted(0.0, CalculateFunction(0.0));
    AddTrialUnsorted(1.0, CalculateFunction(1.0));
    std::sort(trials_.begin(), trials_.end());

    double M = 1.0;
    int k = 1;
    bool stop_flag = false;

    while (k < GetInput().max_iter && !stop_flag) {
        
        // 1. Вычисляем M (одинаково на всех процессах)
        double max_slope = 0.0;
        for (size_t i = 1; i < trials_.size(); ++i) {
            double dx = trials_[i].x - trials_[i-1].x;
            double dz = std::abs(trials_[i].z - trials_[i-1].z);
            if (dx > 1e-15) max_slope = std::max(max_slope, dz / dx);
        }
        if (max_slope > 0) M = GetInput().r * max_slope;
        else M = 1.0;

        // 2. Вычисляем характеристики R для всех интервалов
        std::vector<IntervalInfo> intervals;
        intervals.reserve(trials_.size() - 1);

        for (size_t i = 1; i < trials_.size(); ++i) {
            double dx = trials_[i].x - trials_[i-1].x;
            double z_r = trials_[i].z;
            double z_l = trials_[i-1].z;
            double R = M * dx + ((z_r - z_l) * (z_r - z_l)) / (M * dx) - 2.0 * (z_r + z_l);
            intervals.push_back({R, (int)i});
        }

        // 3. Сортируем интервалы по R (по убыванию) - Правило 4
        std::sort(intervals.begin(), intervals.end(), std::greater<IntervalInfo>());

        // 4. Выбираем top-P интервалов (P = size)
        // Каждый процесс берет на себя обработку одной точки
        double my_new_x = -1.0;
        double my_new_z = 0.0;
        bool i_have_work = false;

        if (rank < intervals.size()) {
            int idx = intervals[rank].index;
            double dx = trials_[idx].x - trials_[idx-1].x;

            if (dx <= GetInput().epsilon) {
                // Если хоть один из лучших интервалов слишком мал, сигнализируем остановку
                // Но в MPI нужно согласовать остановку. 
                // Для простоты: если мой интервал мал, я не генерирую точку, но алгоритм 
                // остановится, когда все лучшие интервалы станут малы.
            } else {
                double x_r = trials_[idx].x;
                double x_l = trials_[idx-1].x;
                double z_r = trials_[idx].z;
                double z_l = trials_[idx-1].z;
                
                double new_x = 0.5 * (x_r + x_l) - (z_r - z_l) / (2.0 * M);
                if (new_x < x_l) new_x = x_l + 1e-9;
                if (new_x > x_r) new_x = x_r - 1e-9;

                my_new_x = new_x;
                my_new_z = CalculateFunction(new_x); // Тяжелая операция
                i_have_work = true;
            }
        }

        // 5. Обмен данными (Allgather)
        // Мы передаем пару {x, z}. Структура из 2 double.
        struct Point { double x; double z; };
        std::vector<Point> local_res(size);
        std::vector<Point> global_res(size);

        if (i_have_work) {
            local_res[rank] = {my_new_x, my_new_z};
        } else {
            local_res[rank] = {-1.0, 0.0}; 
        }
        
        // Используем MPI_Allgather, чтобы собрать результаты со всех процессов
        // Передаем только 1 элемент от себя, но собираем в вектор global_res
        MPI_Allgather(&local_res[rank], 2, MPI_DOUBLE, 
                      global_res.data(), 2, MPI_DOUBLE, MPI_COMM_WORLD);

        // 6. Обновление списка испытаний
        bool points_added = false;
        for (int i = 0; i < size; ++i) {
            if (global_res[i].x >= 0.0) {
                AddTrialUnsorted(global_res[i].x, global_res[i].z);
                points_added = true;
            }
        }

        
        if (!points_added) {
            stop_flag = true;
        } else {
            std::sort(trials_.begin(), trials_.end());
            k += size; 
        }
    }

    double min_z = 1e18;
    double best_t = 0;
    for(const auto& t : trials_) {
        if (t.z < min_z) {
            min_z = t.z;
            best_t = t.x;
        }
    }
    
    double u, v;
    d2xy(best_t, u, v);
    result_.x = GetInput().ax + u * (GetInput().bx - GetInput().ax);
    result_.y = GetInput().ay + v * (GetInput().by - GetInput().ay);
    result_.value = min_z;
    result_.iter_count = k;

    GetOutput() = result_;
    return true;
}

bool KhruevAGlobalOptMPI::PostProcessingImpl() {
    return true;
}

} // namespace khruev_a_global_opt
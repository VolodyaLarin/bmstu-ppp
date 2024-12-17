#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <limits.h>

#include <math.h>
#include <mpi.h>

void measure_metrics(int rank, int size, int matrix_length);

void _to_lu(double *matrix_r, double *inv_matrix, double *rows_getted,
            int matrix_length, int matrix_rows, MPI_Datatype mpi_row, int rank,
            int size);

void _lu_to_identity(double *matrix_r, double *inv_matrix, double *rows_getted,
                     int matrix_length, MPI_Datatype mpi_row, int rank,
                     int size);

void mul_row(double *row, size_t length, double n) {
  assert(row);
  assert(length > 0);

  for (size_t col = 0; col < length; col++) {
    row[col] *= n;
  }
}

void mul_row_mp(double *row, size_t length, double n) {
  assert(row);
  assert(length > 0);
#pragma omp parallel shared(row) firstprivate(length, n)
  {
    size_t col;

#pragma omp for private(col)
    for (col = 0; col < length; col++) {
      row[col] *= n;
    }
  }
}

void mul_matrix(const double *left, const double *right, double *tmp,
                size_t matrix_len) {
  if (!tmp) {
    return;
  }

  memset(tmp, 0, matrix_len * matrix_len * sizeof(double));

  for (size_t i = 0; i < matrix_len; ++i) {
    for (size_t k = 0; k < matrix_len; ++k) {
      for (size_t j = 0; j < matrix_len; ++j) {
        tmp[i * matrix_len + j] +=
            left[i * matrix_len + k] * right[k * matrix_len + j];
      }
    }
  }
}

double inv_check(int matrix_len, double *left, double *right) {
  double *tmp = malloc(sizeof(*tmp) * matrix_len * matrix_len);

  mul_matrix(left, right, tmp, matrix_len);

  double err = 0;
  double diff = 0;

  for (size_t row = 0; row < matrix_len; row++) {
    for (size_t col = 0; col < matrix_len; col++) {
      double data = tmp[row * matrix_len + col];
      double err2;
      if (row == col)
        err2 = fabs(1 - data);
      else
        err2 = fabs(data);

      if (err2 >= err)
        err = err2;
      diff += err2;
    }
  }

  free(tmp);
//   return diff / matrix_len;
    return err;
}

void add_rows(double *row_a, const double *row_b, size_t length, double n) {
  assert(row_a);
  assert(row_b);
  assert(length > 0);

  for (size_t col = 0; col < length; col++) {
    row_a[col] += row_b[col] * n;
  }
}

void add_rows_mp(double *row_a, const double *row_b, size_t length, double n) {
  assert(row_a);
  assert(row_b);
  assert(length > 0);
#pragma omp parallel shared(row_a) firstprivate(row_b, length, n)
  {
    size_t col;
#pragma omp for private(col)
    for (col = 0; col < length; col++) {
      row_a[col] += row_b[col] * n;
    }
  }
}

void print_matrix_2(double *matrix, int rows, int cols, int rank, int size) {
  char sBuffer[1024 * 1024] = {0};

  char *buffer = sBuffer;

  buffer += sprintf(buffer, "===============\n");
  for (int i = 0; i < rows; i++) {
    buffer += sprintf(buffer, "%d :", rank + size * i);
    for (int j = 0; j < cols; j++) {
      buffer += sprintf(buffer, "%+9.5lf ", matrix[i * cols + j]);
    }
    buffer += sprintf(buffer, "\n");
  }
  buffer += sprintf(buffer, "===============\n");

  printf("%s", sBuffer);
}

void print_matrix(double *matrix, int rows, int cols, int rank, int size) {
  printf("===============\n");
  for (int i = 0; i < rows; i++) {
    printf("%d :", rank + size * i);
    for (int j = 0; j < cols; j++) {
      printf("%+9.5lf ", matrix[i * cols + j]);
    }
    printf("\n");
  }
  printf("===============\n");
}

// Задание 22. Разработать программу вычисления матрицы обратной заданной на
// основе метода Р-приведения. Обосновать проектное решение (выбор алгоритма).
// Обеспечить равномерную загрузку процессоров.
// Результат вывести в текстовый файл.
// Исследовать зависимость времени счета от размерности задачи и количества
// процессоров.

void invert_matrix(double *matrix, double *invert_matrix, int matrix_length,
                   int rank, int size) {
  int max_matrix_rows = (matrix_length - 1) / size + 1;
  int matrix_rows = max_matrix_rows;
  if (rank + (matrix_rows - 1) * size >= matrix_length) {
    matrix_rows--;
  }

  double *matrix_r = calloc(sizeof(double), max_matrix_rows * matrix_length);
  double *inv_matrix = calloc(sizeof(double), max_matrix_rows * matrix_length);
  double *rows_getted = calloc(sizeof(double), 2 * matrix_length);

  for (size_t i = 0; i < max_matrix_rows; i++) {
    if (i * size + rank < matrix_length)
      (&inv_matrix[i * matrix_length])[i * size + rank] = 1;
  }

  MPI_Datatype mpi_row;
  MPI_Datatype mpi_row_sh;

  MPI_Type_contiguous(matrix_length, MPI_DOUBLE, &mpi_row);
  {
    MPI_Datatype mpi_row_tmp;
    MPI_Type_vector(max_matrix_rows, matrix_length, matrix_length * size,
                    MPI_DOUBLE, &mpi_row_tmp);
    MPI_Type_create_resized(mpi_row_tmp, 0, sizeof(double) * matrix_length,
                            &mpi_row_sh);
    MPI_Type_free(&mpi_row_tmp);
  }
  MPI_Type_commit(&mpi_row_sh);
  MPI_Type_commit(&mpi_row);

  MPI_Scatter(matrix, 1, mpi_row_sh, matrix_r, max_matrix_rows, mpi_row, 0,
              MPI_COMM_WORLD);

  _to_lu(matrix_r, inv_matrix, rows_getted, matrix_length, matrix_rows, mpi_row,
         rank, size);

  //   print_matrix_2(matrix_r, matrix_rows, matrix_length, rank, size);
  _lu_to_identity(matrix_r, inv_matrix, rows_getted, matrix_length, mpi_row,
                  rank, size);

  free(rows_getted);
  free(matrix_r);

  for (int i = 0; i < max_matrix_rows; i++) {

    MPI_Gather(inv_matrix + matrix_length * i, 1, mpi_row,
               invert_matrix + matrix_length * i * size, 1, mpi_row, 0,
               MPI_COMM_WORLD);
  }
  //    MPI_Gather(inv_matrix, max_matrix_rows, mpi_row,
  //               invert_matrix, 1, mpi_row_sh,
  //               0, MPI_COMM_WORLD);
  //    MPI_Barrier(MPI_COMM_WORLD);
  free(inv_matrix);
  MPI_Type_free(&mpi_row);
  MPI_Type_free(&mpi_row_sh);
}

void _lu_to_identity(double *matrix_r, double *inv_matrix, double *rows_getted,
                     int matrix_length, MPI_Datatype mpi_row, int rank,
                     int size) {
  for (ssize_t i = matrix_length - 1; i > 0; i--) {
    double *row, *irow;
    if (i % size == rank) {
      row = matrix_r + (i / size) * matrix_length;
      irow = inv_matrix + (i / size) * matrix_length;
    } else {
      row = rows_getted;
      irow = rows_getted + matrix_length;
    }

    MPI_Bcast(row, 1, mpi_row, i % size, MPI_COMM_WORLD);
    MPI_Bcast(irow, 1, mpi_row, i % size, MPI_COMM_WORLD);

#pragma omp parallel shared(matrix_r, inv_matrix)                              \
    firstprivate(row, irow, matrix_length)
    {
      size_t j;
#pragma omp for private(j)
      for (j = rank; j < i; j += size) {
        double mul = matrix_r[(j / size) * matrix_length + i];

        add_rows(matrix_r + (j / size) * matrix_length, row, matrix_length,
                 -mul);
        add_rows(inv_matrix + (j / size) * matrix_length, irow, matrix_length,
                 -mul);
      }
    }
  }
}

void _to_lu(double *matrix_r, double *inv_matrix, double *rows_getted,
            int matrix_length, int matrix_rows, MPI_Datatype mpi_row, int rank,
            int size) {
  for (size_t i = 0; i < matrix_length; i++) {
    double *row, *irow;
    if (i % size == rank) {
      row = matrix_r + (i / size) * matrix_length;
      irow = inv_matrix + (i / size) * matrix_length;

      double pivot = row[i];
      mul_row(row, matrix_length, 1 / pivot);
      mul_row(irow, matrix_length, 1 / pivot);
      row[i] = 1;
    } else {
      row = rows_getted;
      irow = rows_getted + matrix_length;
    }
    MPI_Bcast(row, 1, mpi_row, i % size, MPI_COMM_WORLD);
    MPI_Bcast(irow, 1, mpi_row, i % size, MPI_COMM_WORLD);

#pragma omp parallel shared(matrix_r, inv_matrix)                              \
    firstprivate(row, irow, matrix_length)
    {
      size_t j;
#pragma omp for private(j)
      for (j = (i + 1) / size; j < matrix_rows; j++) {
        if (rank + size * j <= i) {
          continue;
        }

        double mul = matrix_r[j * matrix_length + i];
        add_rows(matrix_r + j * matrix_length, row, matrix_length, -mul);
        add_rows(inv_matrix + j * matrix_length, irow, matrix_length, -mul);
      }
    }
  }
}

int main(int argc, char **argv) {
  int rank, size;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (rank == 0) {
    printf("| %10s | %4s | %11s | %2s |\n", "Time", "Size", "Error", "PC");
    printf("|%10s|%4s|%11s|%2s|\n", "------------", "------", "-------------",
           "----");
  }
  int matrix_length = 10;
  if (argc >= 2) {
    for (int i = 1; i < argc; i++) {
      sscanf(argv[i], "%d", &matrix_length);

      if (matrix_length < 1) {
        fprintf(stderr, "Incorrect argument");
        MPI_Finalize();
        return -1;
      }
      measure_metrics(rank, size, matrix_length);
    }
  } else {
    measure_metrics(rank, size, matrix_length);
  }

  MPI_Finalize();
  return 0;
}

void invert_sq_matrix(int matrixsize, double matrix[][matrixsize],
                      double inv_matrix[][matrixsize]) {

  memset(*inv_matrix, 0, matrixsize * matrixsize * sizeof(**inv_matrix));

  for (size_t i = 0; i < matrixsize; i++) {
    inv_matrix[i][i] = 1;
  }

  for (size_t i = 0; i < matrixsize; i++) {
    double pivot = matrix[i][i];
    assert(pivot != 0);

    mul_row(matrix[i], matrixsize, 1 / pivot);
    mul_row(inv_matrix[i], matrixsize, 1 / pivot);
    matrix[i][i] = 1;

    for (size_t j = i + 1; j < matrixsize; j++) {
      double mul = matrix[j][i];
      add_rows_mp(matrix[j], matrix[i], matrixsize, -mul);
      add_rows_mp(inv_matrix[j], inv_matrix[i], matrixsize, -mul);
    }
  }

  for (ssize_t i = matrixsize - 1; i > 0; i--) {
    double pivot = matrix[i][i];
    assert(pivot == 1);
    for (size_t j = 0; j < i; j++) {
      double mul = matrix[j][i];
      add_rows_mp(matrix[j], matrix[i], matrixsize, -mul);
      add_rows_mp(inv_matrix[j], inv_matrix[i], matrixsize, -mul);
    }
  }
}

void measure_metrics(int rank, int size, int matrix_length) {
  double timerStart;
  double *matrix = NULL;
  double *inv_matrix = NULL;

  int max_matrix_rows = (matrix_length - 1) / size + 1;
  if (rank == 0) {

    matrix = calloc(sizeof(*matrix), matrix_length * size * max_matrix_rows);
    for (size_t row = 0; row < matrix_length; row++) {
      for (size_t col = 0; col < matrix_length; col++) {
        matrix[row * matrix_length + col] =
            ((double)rand()) / INT_MAX * 100 - 50;
      }
    }
    inv_matrix =
        calloc(sizeof(*matrix), matrix_length * size * max_matrix_rows);

    timerStart = MPI_Wtime();
  }
  int repeats = 1;

  if (size == 1) {
    for (int i = 0; i < repeats; i++) {
      size_t memSize = matrix_length * size * max_matrix_rows * sizeof(double);
      double *tmp = malloc(memSize);
      memcpy(tmp, matrix, memSize);
      invert_sq_matrix(matrix_length, tmp, inv_matrix);
      free(tmp);
    }
  } else {
    for (int i = 0; i < repeats; i++) {
      invert_matrix(matrix, inv_matrix, matrix_length, rank, size);
    }
  }
  if (rank == 0) {
    double time = MPI_Wtime() - timerStart;

    double err = inv_check(matrix_length, matrix, inv_matrix);
    printf("| %10lf | %4d | %11g | %2d |\n", time / repeats, matrix_length, err,
           size);
    free(matrix);
  }
}

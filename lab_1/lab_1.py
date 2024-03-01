import numpy as np

n = int(input("Square matrix size: "))# размерность единичной матрицы + размерность матрицы A и вектора x
i_ = int(input())#столбец который заменяем
matrix_A = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
    ])
vector_x = np.array([1,0,1])
matrix_A_inverse = 0


try:
    matrix_A_inverse = np.linalg.inv(matrix_A)
    print(f"Обратная матрица:\n{matrix_A_inverse}\n")
except np.linalg.LinAlgError:
    print("Матрица вырожденная, обратной матрицы не существует.")
    exit(0)

for i in range(n):
    matrix_A[i][i_-1] = vector_x[i]

print(matrix_A)
vector_l = matrix_A_inverse @ vector_x

if vector_l[i_-1] != 0:
    l_n_elem = vector_l[i_-1]
    vector_l[i_-1] = -1
    #print(vector_l)
    for i in range(n):
        vector_l[i] = vector_l[i] * (-1 / l_n_elem)

    matrix_q = np.eye(n)

    for i in range(len(matrix_q)):
        matrix_q[i][i_-1] = vector_l[i]
    #print(vector_l)
    matrix_A_dash = matrix_q @ matrix_A_inverse
    #print(matrix_q)
    print(matrix_A_dash)
    print(matrix_A @ matrix_A_dash)#получение единичной матрицы
else:
    print("Матрица не обратима")


#matrix_A = np.array([ пример
   # [2, 1, 0, 0],
   # [3, 2, 0,0],
   # [1, 1, 3,4],
   # [2, -1, 2,3]
   # ])
#vector_x = np.array([7, 8, 9, 10])
import numpy as np

it = 0
def simplex_method(c, x, A, b, B)->list:
    m, n = A.shape
    x_new = x.copy()
    global it
    assert (np.linalg.matrix_rank(A) == m)

    while True:
        # Step 1
        AB = A[:, B]
        A_inv_B = np.linalg.inv(AB)

        # Step 2
        c_B = c[B]

        # Step 3 вектор потенциалов
        u = c_B @ A_inv_B

        # Step 4 вектор оценок
        delta = u @ A - c

        # Step 5
        if np.all(delta >= 0):
            assert np.all(A @ x == b)
            print(it)
            return [x_new, B]

        # Step 6 компонента < 0
        j0 = np.where(delta < 0)[0][0]

        # Step 7
        z = A_inv_B @ A[:, j0]

        # Step 8
        theta = np.array([x_new[B[i]] / z[i] if z[i] > 0 else np.inf for i in range(m)])

        # Step 9
        theta0 = np.min(theta)

        # Step 10
        if theta0 == np.inf:
            raise Exception("Целевая функция не ограничена сверху на множестве допустимых планов")

        # Step 11
        k = np.argmin(theta)
        j_star = B[k]

        # Step 12
        B[k] = j0

        # Step 12
        for i in range(m):
            if i != k:
                x_new[B[i]] -= theta0 * z[i]

        x_new[j0] = theta0
        x_new[j_star] = 0

        it+=1


def start_simplex(A, b):
    it = 0
    while True:
        m, n = A.shape

        # Step 1

        # Form start data
        # Check for negative values
        if not np.all(b >= 0):
            mask = b < 0
            negative_indices = np.where(mask)[0]
            b[mask] *= -1
            A[negative_indices] *= -1

        # Step 2

        c_t_vector = np.zeros(n+m)
        c_t_vector[-m:] = -1

        zero_x_vector = np.zeros(n)
        b_m = b[:m]
        x_vector = np.concatenate((zero_x_vector, b_m))

        zero_matrix = np.eye(m)
        A_vector = np.hstack((A, zero_matrix))

        # Step 3

        B = np.arange(n+1,n+m+1)-1

        # Step 4

        x_vector, B = simplex_method(c_t_vector, x_vector, A_vector, b, B)

        # Step 5

        if np.all(x_vector[-m:] != 0):
            raise Exception("Задача не совместна")

        # Step 6

        x = x_vector[:n]

        assert np.all(A @ x == b)

        # Step 7
        it += 1
        if np.all(np.isin(B,np.arange(0,n))):
            print(it)
            return [x, B, A, b]

        # Step 8

        max_index, k, i = np.max(B), np.argmax(B), np.max(B) - n

        vector_j = np.setdiff1d(np.arange(0, n), B)

        AB = A_vector[:, B]
        A_inv_B = np.linalg.inv(AB)

        l_massive = []

        for j in range(A.shape[1]):
            if j in vector_j:
                A_j = A_vector[:,j]

                multiply = A_inv_B @ A_j

                l_massive.append(multiply)

        l_massive = np.array(l_massive)

        if np.all(l_massive[:, k] != 0):
            not_task = np.where(l_massive[:, k] != 0)[0]
            B[k] = not_task[0]
            continue

        A = np.delete(A, i, axis=0)
        b = np.delete(b,i)
        B = np.delete(B,k)
        A_vector = np.delete(A_vector, i, axis=0)



if __name__ == '__main__':
    c = np.array([1, 0, 0])
    x = np.array([0, 0, 0])
    A = np.array([
        [1, 1, 1],
        [2, 2, 2]
    ])
    b = np.array([0,0])
    B = np.array([0,0])

    print(start_simplex(A, b))



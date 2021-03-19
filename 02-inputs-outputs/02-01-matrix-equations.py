import argparse
import random

import matplotlib.pyplot as plt
import numpy as np
import torch


def element_wise_sum(args, A):
    """
    element wise sum 확인 함수
    :param args: 입력 arguments
    :param A: A matrix
    """
    #
    # B 행렬 합
    #
    B = np.random.randint(-9, 10, (5, 4)) / 10
    B = torch.tensor(B)

    print(A)
    print()
    print(B)
    print()
    print(A + B)

    #
    # B 행렬 Boradcasting 합
    #
    B = np.random.randint(-9, 10, (5, 1)) / 10
    B = torch.tensor(B)

    print(A)
    print()
    print(B)
    print()
    print(A + B)

    # B 행렬 열복사 후 합
    B1 = B.repeat(1, 4)

    print(B1)
    print()
    print(A + B1)

    #
    # B 행렬 Boradcasting 합
    #
    B = np.random.randint(-9, 10, (1, 4)) / 10
    B = torch.tensor(B)

    print(A)
    print()
    print(B)
    print()
    print(A + B)

    # B 행렬 행복사 후 합
    B1 = B.repeat(5, 1)

    print(B1)
    print()
    print(A + B1)

    #
    # B 행렬 Boradcasting 합
    #
    B = np.random.randint(-9, 10, (1, 1)) / 10
    B = torch.tensor(B)

    print(A)
    print()
    print(B)
    print()
    print(A + B)

    # B 행렬 행과 열 복사 후 합
    B1 = B.repeat(5, 4)

    print(B1)
    print()
    print(A + B1)


def element_wise_product(args, A):
    """
    element wise product 확인 함수
    :param args: 입력 arguments
    :param A: A matrix
    """
    #
    # B 행렬 element wise product
    #
    B = np.random.randint(-9, 10, (5, 4)) / 10
    B = torch.tensor(B)

    print(A)
    print()
    print(B)
    print()
    print(A * B)

    #
    # B 행렬 Boradcasting element wise product
    #
    B = np.random.randint(-9, 10, (5, 1)) / 10
    B = torch.tensor(B)

    print(A)
    print()
    print(B)
    print()
    print(A * B)

    # B 행렬 열복사 후 element wise product
    B1 = B.repeat(1, 4)

    print(B1)
    print()
    print(A * B1)

    #
    # B 행렬 Boradcasting element wise product
    #
    B = np.random.randint(-9, 10, (1, 4)) / 10
    B = torch.tensor(B)

    print(A)
    print()
    print(B)
    print()
    print(A * B)

    # B 행렬 행복사 후 element wise product
    B1 = B.repeat(5, 1)

    print(B1)
    print()
    print(A * B1)

    #
    # B 행렬 Boradcasting element wise product
    #
    B = np.random.randint(-9, 10, (1, 1)) / 10
    B = torch.tensor(B)

    print(A)
    print()
    print(B)
    print()
    print(A * B)

    # B 행렬 행과 열 복사 후 합
    B1 = B.repeat(5, 4)

    print(B1)
    print()
    print(A + B1)


def matrix_multiplication(args, A):
    """
    matrix multiplication 확인 함수
    :param args: 입력 arguments
    :param A: A matrix
    """
    #
    # B 행렬 matrix multiplication
    #
    B = np.random.randint(-9, 10, (4, 3)) / 10
    B = torch.tensor(B)

    print(A)
    print()
    print(B)
    print()
    print(torch.matmul(A, B))

    # 직접계산
    result = torch.zeros(A.size(0), B.size(-1))
    for row in range(A.size(0)):
        for col in range(B.size(-1)):
            result[row][col] = torch.dot(A[row], B[:, col])

    print(result)


def dot_product(args):
    """
    dot product 확인 함수
    :param args: 입력 arguments
    """
    # a, b 벡터 선언
    a = np.random.randint(-9, 10, (5,)) / 10
    a = torch.tensor(a)

    b = np.random.randint(-9, 10, (5,)) / 10
    b = torch.tensor(b)

    print(a)
    print()
    print(b)

    # dot-product
    print(torch.dot(a, b))

    # element wise product and sum
    c = a * b

    print(c)
    print()
    print(torch.sum(c))


def main(args):
    """
    동작을 실행하는 main 함수
    :param args: 입력 arguments
    """
    A = np.random.randint(-9, 10, (5, 4)) / 10
    A = torch.tensor(A)
    print(A)

    element_wise_sum(args, A)
    element_wise_product(args, A)
    matrix_multiplication(args, A)
    dot_product(args)


def set_seed(seed):
    """ random seed 설정 """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    """ 동작에 필요한 arguments 설정 """
    parser = argparse.ArgumentParser(description="Matrix Equations arguments.")

    parser.add_argument("--seed", default=1234, type=int, help="random seed value")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    # CPU 또는 GPU 사용여부 결정
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main(args)

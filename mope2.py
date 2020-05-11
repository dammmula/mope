import math
import numpy as np
x1_min = 20
x1_max = 70
x2_min = 25
x2_max = 65
m = 5

first = [-1, -1, 9, 10, 11, 15, 9]
second = [1, -1, 15, 14, 10, 12, 14]
third = [-1, 1, 20, 18, 12, 10, 16]

def commonValue(list):
    total = 0
    for i in range(2,7):
        total += list[i]
    return total/5

#середні значення
commonFirst = commonValue(first)
commonSecond = commonValue(second)
commonThird  = commonValue(third)

def disspersion(commonValue, list):
    total = 0
    for i in range(2,7):
        total += math.pow((list[i]-commonValue),2)
    return total/5

#дисперсії
dispersionFirst = disspersion(commonFirst, first)
dispersionSecond = disspersion(commonSecond, second)
disspersionThird = disspersion(commonThird, third)

dispersionFirst = 0.53
dispersionSecond = 0.53
disspersionThird = 1.24

#відхилення
deviation = math.sqrt((4*m-4)/(m*(m-4)))

Fuv_1 = dispersionFirst/dispersionSecond
Fuv_2 = disspersionThird/dispersionFirst
Fuv_3 = disspersionThird/dispersionSecond

teta_1  = (m-2) * Fuv_1 /m
teta_2  = (m-2) * Fuv_2 /m
teta_3  = (m-2) * Fuv_3 /m

#експериментальне Романовського (однорідність дисперсії)
Ruv_1 = math.fabs(teta_1-1)/deviation
Ruv_2 = math.fabs(teta_2-1)/deviation
Ruv_3 = math.fabs(teta_3-1)/deviation

#нормовані коефіцієнти
mx1 = (first[0] + second[0] + third[0])/3
mx2 = (first[1] + second[1] + third[1])/3
my = (commonFirst + commonSecond + commonThird)/3

a1 = (first[0]**2 + second[0]**2 + third[0]**2)/3
a2 = (first[0]*first[1] + second[0]*second[1] + third[0]*third[1])/3
a3 = (first[1]**2 + second[1]**2 + third[1]**2)/3

a11 = (first[0] * commonFirst + second[0] * commonSecond + third[0] * commonThird)/3
a22 = (first[1] * commonFirst + second[1] * commonSecond + third[1] * commonThird)/3

b0 = np.linalg.det(np.array([
              [my, mx1, mx2],
              [a11, a1, a2],
              [a22, a2, a3]]))/np.linalg.det(np.array([
              [1, mx1, mx2],
              [mx1, a1, a2],
              [mx2, a2, a3]]))
b1 = np.linalg.det(np.array([
              [1, my, mx2],
              [mx1, a11, a2],
              [mx2, a22, a3]]))/np.linalg.det(np.array([
              [1, mx1, mx2],
              [mx1, a1, a2],
              [mx2, a2, a3]]))
b2 = np.linalg.det(np.array([
              [1, mx1, my],
              [mx1, a1, a11],
              [mx2, a2, a22]]))/np.linalg.det(np.array([
              [1, mx1, mx2],
              [mx1, a1, a2],
              [mx2, a2, a3]]))
print(f"y = {round(b0,2)} + {round(b1, 2)}*x1 + {round(b2,2)}*x2")
print(b0-1*b1-1*b2)


#натуралізація рівняння регресії
dx1 = math.fabs(x1_max-x1_min)/2
dx2 = math.fabs(x2_max-x2_min)/2
x10 = (x1_max + x1_min)/2
x20 = (x2_max + x2_min)/2

a0 = b0 - b1*x10/dx1 -b2*x20/dx2
a1 = b1/dx1
a2 = b2/dx2
print(f"naturalized\ny = {round(a0,2)} + {round(a1,2)}*x1 + {round(a2,2)}*x2")

import matplotlib.pyplot as plt
import numpy as np

""""среднее значение величины"""
def mean_value(X: list[float]) -> float:
    summ=0
    for x in X:
        summ += x
    mean = summ/len(X)
    return mean

""""среднее квадратичное"""
def mean_value2(X: list[float]) -> float:
    summ=0
    for x in X:
        summ += x**2
    mean = summ/(len(X)-1)
    return mean

""""среднее от произведения"""
def mean_XY(X:list[float],Y:list[float]) -> float:
    summXY = 0
    for x in range(len(X)):
        summXY += X[x]*Y[x]
    meanXY=summXY/(len(X)-1)
    return meanXY


"""погрешность значения"""
def local_mistake(X: list[float]) -> float:
    summ_sqr=0
    for x in X:
        summ_sqr +=(x-mean_value(X))**2
    localmistake = (summ_sqr/(len(X)-1))**0.5
    return localmistake


"""погрешность среднего значения"""
def mean_mistake(X: list[float]) -> float:
    summ_sqr=0
    for x in X:
        summ_sqr +=(x-mean_value(X))**2
    meanmistake = summ_sqr**0.5/(len(X)-1)
    return meanmistake


""""среднее значение величины"""
def mean_value(X: list[float]) -> float:
    summ=0
    for x in X:
        summ += x
    mean = summ/len(X)
    return mean

"""ищем угловой коофициент наклонной, а так же ее погрешность линейность"""
def MNK_linear(X:list[float],Y:list[float])-> float:
    k = ( mean_XY(X,Y) - mean_value(X) * mean_value(Y) ) / ( mean_value2(X) - mean_value(X)**2 )
    errk2 = ( (mean_value2(Y)-mean_value(Y)**2)  / (mean_value2(X)-mean_value(X)**2)  - k**2 ) / ( len(X) - 2 )
    b = mean_value(Y) - k * mean_value(X)
    errb = errk2**0.5 * mean_value2(X)**0.5
    return k, errk2**0.5, b,errb

def MNK_proprtional(X:list[float],Y:list[float])-> float:
    k = mean_XY(X,Y)/mean_value2(X)
    errk2 = (mean_value2(Y)/mean_value2(X)-k**2)/(len(X)-1)
    return k, errk2**0.5

f_cup =[3159,	6315,	9488,	12638,	15817,	18992,	22120]
f_all = [4262,	8544,	12785,	17048,	21297]
f_stell = [4125,	8257,	12381,	16514,	20632]
n1 = [1,2,3,4,5,6,7]
n2 = [1,2,3,4,5]

Xapprox1 = np.linspace(0, 7.3, 2)
Xapprox2 = np.linspace(0, 5.3, 2)

Yapprox1 = []
Yapprox2 = []
Yapprox3 = []
for x in Xapprox1:
    Yapprox1.append(MNK_proprtional(n1, f_cup)[0]*x)

for x in Xapprox2:
    Yapprox2.append(MNK_proprtional(n2, f_all)[0]*x)

for x in Xapprox2:
    Yapprox3.append(MNK_proprtional(n2, f_stell)[0]*x)

#plt.plot(Xapprox1, Yapprox1,color="red",label="медь")
plt.errorbar(n1, f_cup, xerr=0, yerr=0, fmt=".", color="black",markersize=7)

plt.plot(Xapprox2, Yapprox2,color="green",label="дюралюминий")
plt.errorbar(n2, f_all, xerr=0, yerr=0, fmt=".", color="black",markersize=7)


plt.plot(Xapprox2, Yapprox3,color="blue",label="сталь")
plt.errorbar(n2, f_stell, xerr=0, yerr=0, fmt=".", color="black",markersize=7)

# косметика
plt.xlabel('n')
plt.ylabel("f_n, Гц")
plt.grid()
plt.title("Зависимость частоты колебаний от номера гармоники")
plt.legend(fontsize='10')

plt.show()



print(MNK_proprtional(n1, f_cup))
print(MNK_proprtional(n2, f_all))
print(MNK_proprtional(n2, f_stell))
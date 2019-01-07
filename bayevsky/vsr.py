import math
import numpy as np
import matplotlib.pyplot as plt
from numpy import median
from scipy.interpolate import interp1d

delete_indexes = []


def get_bayevsky(rr_list, fd=4.0):
    if not(type(rr_list)) is list:
        raise ValueError("RR intervals not a list")
    if len(rr_list) == 0:
        raise ValueError("List of RR intervals is empty")
    for i in range(len(rr_list)):
            if not(rr_list[i].isdigit()):
                raise ValueError("Element of RR ", rr_list[i], " list is not a number")
    rr_list = np.array(rr_list, dtype=np.int)
    rr_list = cleaner(rr_list)
    rr_list = recovery(rr_list, fd)
    create_ritmogramm(rr_list)
    ampl = furye_change(rr_list)
    vsr_results = calculation(rr_list, ampl)
    plt.show()
    return vsr_results


def cleaner(rr_list):
    # Первый этап очистки 0,15% медианы 10 чисел
    pointer = 0
    while pointer + 1 <= len(rr_list):
        current_ten = np.array(rr_list[pointer:pointer + 10])  # Выбираем 10 элементов
        mediana = median((current_ten))
        for i in range(len(current_ten)):
            if (float(current_ten[i]) > mediana * 1.15) or (float(current_ten[i]) < mediana * 0.85):
                #print("Element ", current_ten[i]," > ", mediana*1.15, " or < ", mediana*0.85)
                delete_indexes.append(i+pointer)
        pointer += 10

    # Второй этап процесса очистки 0,25% медианы 20 чисел
    pointer = 0
    while pointer + 1 <= len(rr_list):
        current_ten = np.array(rr_list[pointer:pointer + 20])  # Выбираем 20 элементов
        mediana = median((current_ten))
        for i in range(len(current_ten)):
            if (float(current_ten[i]) > mediana * 1.15) or (float(current_ten[i]) < mediana * 0.85):
                #print("Element ", current_ten[i]," > ", mediana*1.15, " or < ", mediana*0.85)
                delete_indexes.append(i+pointer)
        pointer += 20
    return np.delete(rr_list, delete_indexes)


def recovery(rr_list, fd=4.0):
    # Линейнвя и кубическая интерполяция
    uniq_del_ind = list(set(delete_indexes))
    uniq_del_ind.sort()
    #print("Del_ind_sort", uniq_del_ind)
    #print("Len: ", len(uniq_del_ind))
    k=0
    for i in range(len(uniq_del_ind)):
        uniq_del_ind[i] = uniq_del_ind[i] - i
    for ind in range(len(uniq_del_ind)):
        #print("Длина списка", len(rr_list))
       # print("Количество вставленных", k)
        if (uniq_del_ind[ind]!= 0) and (uniq_del_ind[ind] != len(rr_list)):
            elem = int((rr_list[uniq_del_ind[ind]] + rr_list[uniq_del_ind[ind] - 1]) / 2)
            rr_list = np.insert(rr_list, uniq_del_ind[ind], elem)
            k+=1
            for k in range(len(uniq_del_ind)):
                uniq_del_ind[k] += 1
        elif uniq_del_ind[ind] == 0:
            elem = int((rr_list[0] + rr_list[1]) / 2)
            rr_list = np.insert(rr_list, uniq_del_ind[ind], elem)
            k+=1
            for k in range(len(uniq_del_ind)):
                uniq_del_ind[k] += 1
        elif uniq_del_ind[ind] == len(rr_list):
            elem = int((rr_list[uniq_del_ind[ind]-1] + rr_list[uniq_del_ind[ind] - 2]) / 2)
            rr_list = np.insert(rr_list, uniq_del_ind[ind], elem)
            k += 1
            for k in range(len(uniq_del_ind)):
                uniq_del_ind[k] += 1
    #print("rr_list_final with length ", len(rr_list))
    rr_x = np.arange(len(rr_list))
    x = np.arange(rr_x.min(), rr_x.max(), 1 / fd)  # 1/fd частота равномерной дискритизации
    f = interp1d(rr_x, rr_list, kind='cubic')
    rr_list_interp = f(x)  # получение интерполированной функции с заданной частотой
    return rr_list_interp


def create_ritmogramm(rr_list):
    # Отрисовка ритмограммы
    ax = plt.subplot(3, 1, 1)
    ax.set_ylim(min(rr_list) - 50, max(rr_list) + 50)
    ax.set_xlim(0, len(rr_list))
    ax.fill_between(range(len(rr_list)), rr_list)
    #ax.set_title("Ритмограмма")
    #ax.set_ylabel('Величина RR, мс')
    #ax.set_xlabel("N, шт.")
    plt.grid()
    plt.plot(rr_list)


def furye_change(rr_list):
    # Преобразование Фурье и получение значений спектра
    ax3 = plt.subplot(3, 1, 2)
    ax3.grid(linestyle='-', color='blue')
    #ax3.set_title("Спектрограмма")
    #ax3.set_ylabel('Амплитуда, мс^2/Гц * 10^4')
    #ax3.set_xlabel("Частота, Гц")
    furye = np.fft.rfft(rr_list)
    frequency = np.fft.rfftfreq(len(rr_list), d=1.25)  # До 0.4 Гц частота
    ax3.set_xlim(frequency[1], 0.45)
    amplitude = np.abs(furye)
    ax3.set_ylim(0, max(amplitude[2:]) / math.pow(10, 4))
    plt.grid()
    plt.plot(frequency, (amplitude / math.pow(10, 4)))
    return amplitude


def calculation(rr_list, amplitude):
    # Функция расчета параментов ВСР
    # 1) Построение гистограммы распределения интервалов
    bins = np.linspace(300, 1700, int((1700 - 300) / 50) + 1)
    ax1 = plt.subplot(3, 1, 3)
    #ax1.set_title("Гистограмма распределения RR-интервалов")
    #ax1.set_xlabel("T, сек")
    #ax1.set_ylabel("N, шт.")
    rr_hist = list(plt.hist(rr_list, bins, color='green', edgecolor='black'))  # Гистограмма распределения RR-интервалов

    # 2) Рассчет статических параметров
    AMo = max(rr_hist[0])/len(rr_list)
    max_ind = rr_hist[0].argmax()
    if int(rr_hist[1][max_ind]) <= 1650:
        Mo = (rr_hist[1][max_ind] + (rr_hist[0][max_ind]-rr_hist[0][max_ind-1])*50/(2*rr_hist[0][max_ind]-rr_hist[0][max_ind-1]-rr_hist[0][max_ind+1]))
    else:
        Mo = (1675)
    #print("AMo = ", AMo, " Mo = ", Mo)
    MxDMn = max(rr_list) - min(rr_list)  # Вариационный размах
    SI = (AMo * 100)/(2 * Mo * MxDMn*pow(10, -6))
    VH = 0
    if SI >= 500:
        VH = 2
    elif (SI > 25) and (SI <= 50):
        VH = -1
    elif (SI > 50) and (SI < 200):
        VH = 0
    elif (SI < 500) and (SI >= 200):
        VH = 1
    elif SI <= 25:
        VH = -2
    NN_average = sum(rr_list)/len(rr_list)  # Стресс-индекс
    HR = (1000/NN_average)*60
    TER = 0
    if (HR <= 50):
        TER = -2
        #print("Выраженная брадикардия")
    elif (HR > 50 and HR <= 60):
        TER = -1
        #print("Умеренная брадикардия")
    elif (HR > 60 and HR < 75):
        TER = 0
        #print("Нормокардия")
    elif (HR >= 75 and HR < 90):
        TER = 1
        #print("Умеренная тахикардия")
    elif (HR >= 90):
        TER = 2
        #print("Выраженная тахикардия")
    SDNN = np.std(rr_list, dtype='double')
    if SDNN >= 600:
        FA = -2
    elif SDNN>=450 and SDNN<600:
        FA = -1
    elif(SDNN>300 and SDNN<450):
        FA = 0
    elif(SDNN<=300 and SDNN>100):
        FA = -1
    elif(SDNN<=100):
        FA = -2
    CV = (SDNN / NN_average)*100
    if(CV >= 6):
        SR = -2
    elif(CV >3 and CV < 6):
        SR = 0
    elif(CV <= 3):
        SR = 2
    NN50 = 0
    for i in range(len(rr_list)-1):
        if abs((rr_list[i+1] - rr_list[i])>=50):
            NN50+=1
    N_ar = 0
    for i in range(len(rr_list)):
        x = 100*rr_list[i]/NN_average
        if abs(x-100)>10:
            N_ar += 1
    NArr = N_ar*100/len(rr_list)
    pNN50 = NN50*100/len(rr_list)

    # 3) Рассчет частотных показателей
    frequency = np.fft.rfftfreq(len(rr_list), d=1.25)
    TP = float(np.trapz(amplitude, frequency))
    print("TP = ", TP)
    vlf_start_index = 0
    lf_start_index = 0
    hf_start_index = 0
    for i in range(len(frequency)):
        if( i==0 ):
            pass
        elif(frequency[i]>=0.003 and frequency[i-1]<0.003):
            vlf_start_index = i
        elif(frequency[i]>=0.04 and frequency[i-1]<0.04):
            lf_start_index = i
        elif(frequency[i]>=0.15 and frequency[i-1]<0.15):
            hf_start_index = i
    VLF = float(np.trapz(amplitude[vlf_start_index:lf_start_index-1], frequency[vlf_start_index:lf_start_index-1]))
    ASNC = 0
    if (VLF/TP)*100 <= 20:
        ASNC = -2
    elif ((VLF / TP) * 100 <= 40) and ((VLF / TP) * 100 > 20):
        ASNC = -1
    elif ((VLF / TP) * 100 < 60) and ((VLF / TP) * 100 > 40):
        ASNC = 0
    elif ((VLF / TP) * 100 < 70) and ((VLF / TP) * 100 >= 60):
        ASNC = 1
    elif (VLF / TP) * 100 >= 70:
        ASNC = 2
    LF = float(np.trapz(amplitude[lf_start_index:hf_start_index - 1], frequency[lf_start_index:hf_start_index - 1]))
    HF = float(np.trapz(amplitude[hf_start_index:], frequency[hf_start_index:]))
    IC = (HF + LF)/ VLF
    IRSA = math.fabs(TER + FA + VH + SR + ASNC)
    vsr_results = {'HR': HR, 'SDNN': SDNN, 'CV': CV, 'SI': SI, 'NArr': NArr,
    'NN50': NN50, 'pNN50': pNN50, 'TP': TP, 'VLF': VLF, 'LF': LF, 'HF': HF, 'VLF%': (VLF/TP)*100, 'LF%': LF*100/(LF+HF),
    'HF%': HF*100/(LF+HF), 'LF/HF': LF/HF, 'IC': IC, 'IRSA': IRSA}
    #print(vsr_results)
    return vsr_results

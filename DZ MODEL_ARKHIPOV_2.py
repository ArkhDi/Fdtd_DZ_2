# -*- coding: utf-8 -*-

import numpy as np
from numpy.fft import fft, fftshift
import tools
import matplotlib.pyplot as plt

class GaussianDiff:
    '''
    Источник, создающий дифференцированный гауссов импульс
    '''

    def __init__(self, magnitude, A_max, F_max,dt,Sc=1.0,eps=1.0,mu=1.0,):
        '''
        magnitude - максимальное значение в источнике;
        dg - коэффициент, задающий начальную задержку гауссова импульса;
        wg - коэффициент, задающий ширину гауссова импульса.
        '''
        self.magnitude = magnitude
        
        self.Sc = Sc
        self.dt=dt
        self.eps = eps
        self.mu = mu
        self.A_max = A_max
        self.F_max = F_max

    def getE(self, m, q):
        w_g = np.sqrt(np.log(5.5 * self.A_max)) / (np.pi * self.F_max)
        d_g = w_g * np.sqrt(np.log(2.5 * self.A_max * np.sqrt(np.log(2.5 * self.A_max))))
        e = ((q - m * np.sqrt(self.eps * self.mu) / self.Sc) - d_g/dt) / (w_g/dt)
        return -2 * self.magnitude * e * np.exp(-(e ** 2))

if __name__ == '__main__':

    d = 1e-3
    # Характеристическое сопротивление свободного пространства
    Z0 = 120.0 * np.pi

    # Число Куранта
    Sc = 1.0

    # Магнитная постоянная
    mu0 = np.pi * 4e-7

    # Электрическая постоянная
    eps0 = 8.854187817e-12

    # Скорость света в вакууме
    c = 1.0 / np.sqrt(mu0 * eps0)
    # Расчет "дискретных" параметров моделирования
    Cdtds = 1.0 / np.sqrt(2.0)

    dt = d / c * Cdtds
    print (dt)
    # Время расчета в отсчетах
    maxTime_sec = 6e-9
    maxTime = int(np.ceil(maxTime_sec / dt))
    
    # Размер области моделирования в отсчетах
    sizeX_m = 1
    maxSize = int(np.ceil(sizeX_m / d))
    layer_x = 0.6
    layer_x_DX=int(np.ceil(layer_x / d))
    layer_x_2=0.6+0.02
    layer_x_2_DX=int(np.ceil(layer_x_2 / d))
    layer_x_3=0.6+0.02+0.04
    layer_x_3_DX=int(np.ceil(layer_x_3 / d))

    # Положение источника в отсчетах
    sourcePos = 0.3
    sourcePos = int(np.ceil(sourcePos / d))

    # Датчики для регистрации поля
    probesPos = [int(np.ceil(0.4 / d)),int(np.ceil(0.15 / d))]
    
    probes = [tools.Probe(pos, maxTime) for pos in probesPos]

    # Параметры среды
    eps = np.ones(maxSize)
    eps[layer_x_DX:] = 2
    eps[layer_x_2_DX:] = 6.8
    eps[layer_x_3_DX:] = 1.5
    mu = np.ones(maxSize-1)
    
    Ez = np.zeros(maxSize)
    Ezspectrumpad = np.zeros(maxSize)
    Ezspectrumotr = np.zeros(maxSize)
    Hy = np.zeros(maxSize - 1)

    source = GaussianDiff(1,5000, 20e9, dt)

    #Коэффициенты для расчета ABC второй степени
    # Sc' для правой границы
    Sc1Right = Sc / np.sqrt(mu[-1] * eps[-1])

    k1Right = -1 / (1 / Sc1Right + 2 + Sc1Right)
    k2Right = 1 / Sc1Right - 2 + Sc1Right
    k3Right = 2 * (Sc1Right - 1 / Sc1Right)
    k4Right = 4 * (1 / Sc1Right + Sc1Right)

    # Ez[0: 2] в предыдущий момент времени (q)
    oldEzLeft1 = np.zeros(3)

    # Ez[0: 2] в пред-предыдущий момент времени (q - 1)
    oldEzLeft2 = np.zeros(3)

    # Ez[-3: -1] в предыдущий момент времени (q)
    oldEzRight1 = np.zeros(3)

    # Ez[-3: -1] в пред-предыдущий момент времени (q - 1)
    oldEzRight2 = np.zeros(3)

    # Параметры отображения поля E
    display_field = Ez
    display_ylabel = 'Ez, В/м'
    display_ymin = -1.1
    display_ymax = 1.1

    # Создание экземпляра класса для отображения
    # распределения поля в пространстве
    display = tools.AnimateFieldDisplay(maxSize,
                                        display_ymin, display_ymax,
                                        display_ylabel)

    display.activate()
    display.drawProbes(probesPos)
    display.drawSources([sourcePos])
    display.drawBoundary(layer_x_DX)
    display.drawBoundary(layer_x_2_DX)
    display.drawBoundary(layer_x_3_DX)
    

    for q in range(1, maxTime):
        # Расчет компоненты поля H
        Hy[:] = Hy + (Ez[1:] - Ez[:-1]) * Sc / (Z0 * mu)

        # Источник возбуждения с использованием метода
        # Total Field / Scattered Field
        Hy[sourcePos - 1] -= Sc / (Z0 * mu[sourcePos - 1]) * source.getE(0, q)

        # Расчет компоненты поля E
        Ez[1:-1] = Ez[1:-1] + (Hy[1:] - Hy[:-1]) * Sc * Z0 / eps[1:-1]

        # Источник возбуждения с использованием метода
        # Total Field / Scattered Field
        Ez[sourcePos] += (Sc / (np.sqrt(eps[sourcePos] * mu[sourcePos])) *
                          source.getE(-0.5, q + 0.5))

        Sc1Left = Sc / np.sqrt(mu[0] * eps[0])
        k1Left = -1 / (1 / Sc1Left + 2 + Sc1Left)
        k2Left = 1 / Sc1Left - 2 + Sc1Left
        k3Left = 2 * (Sc1Left - 1 / Sc1Left)
        k4Left = 4 * (1 / Sc1Left + Sc1Left)
        # Граничные условия ABC второй степени (слева)
        Ez[0] = (k1Left * (k2Left * (Ez[2] + oldEzLeft2[0]) +
                           k3Left * (oldEzLeft1[0] + oldEzLeft1[2] - Ez[1] - oldEzLeft2[1]) -
                           k4Left * oldEzLeft1[1]) - oldEzLeft2[2])

        oldEzLeft2[:] = oldEzLeft1[:]
        oldEzLeft1[:] = Ez[0: 3]
        # Граничные условия ABC второй степени (справа)
        Ez[-1] = (k1Right * (k2Right * (Ez[-3] + oldEzRight2[-1]) +
                             k3Right * (oldEzRight1[-1] + oldEzRight1[-3] - Ez[-2] - oldEzRight2[-2]) -
                             k4Right * oldEzRight1[-2]) - oldEzRight2[-3])

        oldEzRight2[:] = oldEzRight1[:]
        oldEzRight1[:] = Ez[-3:]

        # Регистрация поля в датчиках
        for probe in probes:
            probe.addData(Ez, Hy)
          


        if q % 30 == 0:
            display.updateData(display_field, q)


    display.stop()

    # Отображение сигналов, сохраненных в датчиках
    tools.showProbeSignals(probes, -1.1, 1.1)
    
    # Расчет спектра

    size = maxSize

    df = 1 / (size * dt)
    print(df)
    start_index=100
    for q in range(1, maxSize):
        Ezspectrumpad[q]=probes[0].E[q]
        #Ezspectrumpad[start_index:] = [0] * (len(Ezspectrumpad) - start_index)
        Ezspectrumotr[q]=probes[1].E[q]
    spectrumpad = np.abs(fft(Ezspectrumpad))
    spectrumpad = fftshift(spectrumpad)
    
    spectrumotr = np.abs(fft(Ezspectrumotr))
    spectrumotr = fftshift(spectrumotr)

    # Расчет частоты
    freq = np.arange(-size / 2 * df, size / 2 * df, df)
    # Отображение спектра
    plt.subplot(1, 1, 1)
    plt.plot(freq, spectrumpad / np.max(spectrumpad))
    plt.plot(freq, spectrumotr / np.max(spectrumotr))
    plt.grid()
    plt.xlabel('Частота, Гц')
    plt.ylabel('|S| / |Smax|')
    plt.xlim(7.5e9, 20e9)

    koefotr=spectrumotr/(spectrumpad)
    

    plt.subplot(2, 2, 2)
    plt.plot(freq, koefotr)
    plt.grid()
    plt.ylabel('Г')
    
    plt.xlim(7.5e9, 20e9)

    plt.subplots_adjust(wspace=0.4)
    plt.show()


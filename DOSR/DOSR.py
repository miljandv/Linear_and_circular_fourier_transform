import numpy as np
import matplotlib.pyplot as plt
from scipy import array, zeros, signal
from scipy.fftpack import fft, ifft, convolve
import math
import scipy.io.wavfile as wav
import soundfile
import wave
import librosa


gggg = 2017
bbbb = 264
P = bbbb % 4
Q = (gggg*(0+2+6+4)) % 4
R = (bbbb*(2+0+1+7)) % 4
S = (bbbb + gggg) % 4
print('P= ', P)
print('Q= ', Q)
print('R= ', R)
print('S= ', S)
N = 10*(P+1)
print('N= ', N)
f_start = 4.4
if Q!=0:
    fs = f_start * (10*Q)
else:
    fs = f_start

def f_1():
    global P
    global Q
    global R
    global S
    global N
    draw_linear_and_circular_conv_results()


def func_x(n,N):
    if( 0 <= n <= (N/2-1)):
        return n
    elif (N/2 <= n <= N-1):
        ret = 1-n-N
        return ret
    else:
        return 0


def func_y(n,N):
    global P
    if (0 <= n <= N/2-1):
        ret = 2*np.cos((P+1)*n+np.pi/4)
        return ret
    else:
        return 0


def idft(t):
    res = np.zeros(len(t))
    for k in range(0, len(t)):
        val = 0.0
        for n in range(0, len(t)):
            val += t[n]*np.exp(1.j * 2*np.pi * n * k / len(t))
        res[k] = val.real
    return res


def ccirc(x,y):
    res = fft(x)*fft(y)
    ret = ifft(res)
    return ret 


def lconv(x,y):
    conv_len = len(x)+len(y)-1
    x_pad = np.pad(x,(0,conv_len-len(x)),'constant')
    y_pad = np.pad(y,(0,conv_len-len(y)),'constant')
    res = fft(x_pad)*fft(y_pad)
    ret = ifft(res)
    return ret


def draw_linear_and_circular_conv_results():
    global N
    y = []
    x = []
    x_prod_y = []
    xdraw = []
    ydraw = []
    x_draw = np.linspace(-1.5,12.5, 90)
    x_ = np.linspace(-1.5,12.5, int(N/2))
    x_c = np.linspace(-1.5,5.5, int(N/2))
    for i in x_draw:
        xdraw.append(func_x(i,N))
        ydraw.append(func_y(i,N))
    for i in x_:
        xapp = (func_x(i,N))
        yapp = (func_y(i,N))
        x.append(xapp)
        y.append(yapp)
        x_prod_y.append(xapp*yapp)
    x_inv_disc_four = idft(x)
    y_inv_disc_four = idft(y)
    x_conv_y = np.convolve(x,y,mode='full')
    x_prod_y_inv_disc_four = idft(x_prod_y)

    
    fig1, ((ax1,ax2)) = plt.subplots(1, 2,sharex=True)
    fig1.suptitle('Grafici funkcija x[n] i y[n]')
    ax1.stem(x_draw, xdraw, use_line_collection = True)
    ax2.stem(x_draw, ydraw, 'tab:orange', use_line_collection = True)
    ax1.title.set_text('Signal x[n]')
    ax1.set_xlabel("n")
    ax2.set_xlabel("n")
    ax1.set_ylabel("x[n]")
    ax2.set_ylabel("y[n]")
    ax2.title.set_text('Signal y[n]')
    plt.show()


    x_conv = np.linspace(-1.5,12.5, int(N)-1)
    fig2 = plt.figure()
    plt.stem(x_conv, x_conv_y, use_line_collection=True)
    plt.title('Linear convolution of x and y')
    plt.xlabel('n')
    plt.ylabel('x_conv_y[n]')
    plt.draw()
    plt.waitforbuttonpress(0)
    plt.close(fig2)

    x_conv_y_circ = ccirc(x,y)
    fig3 = plt.figure()
    plt.stem(x_c, x_conv_y_circ, use_line_collection=True)
    plt.title('Circular convolution of x and y')
    plt.xlabel('n')
    plt.ylabel('x_cconv_y[n]')
    plt.draw()
    plt.waitforbuttonpress(0)
    plt.close(fig3)

    
    x_conv_y_circ_padded = lconv(x,y)
    fig4 = plt.figure()
    plt.stem(x_conv, x_conv_y_circ_padded, use_line_collection=True)
    plt.title('Padded Circular convolution so it equals Linear convolution of x and y')
    plt.xlabel('n')
    plt.ylabel('x_cconv_y[n]')
    plt.draw()
    plt.waitforbuttonpress(0)
    plt.close(fig4)



def f_2():
    global P
    global Q
    global R
    global S
    global N
    global fs
    fs,signal=wav.read("C:/Users/milja/source/repos/DOSR/DOSR/data/audio"+str(Q)+".wav")
    #signal.setflags(write=1)
    #signal /= 32767
    audio = []
    print(signal.size)
    for i in range(signal.size):
        audio.append(signal[i]/32767)
    fig1 = plt.figure()
    plt.title('Audio at 44100Hz')
    plt.plot(audio)
    plt.waitforbuttonpress(0)
    plt.draw()
    plt.close(fig1)

    y, s = librosa.load("C:/Users/milja/source/repos/DOSR/DOSR/data/audio"+str(Q)+".wav", sr=4.4)
    fig2 = plt.figure()
    plt.title('Audio at '+str(fs)+'Hz')
    plt.plot(audio)
    plt.waitforbuttonpress(0)
    plt.draw()
    plt.close(fig2)



def main():
   f_1()


if __name__ == '__main__':
    main()











# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 12:10:09 2024

@author: 08sha
"""
# importing libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
from scipy.fft import fft, ifft, fftfreq
from scipy import signal


def powspec(wave, N, T):
    '''
    Calculates power spectrum and its frequency from given signal input
    '''
    ft = fft(wave)
    power = (np.abs(ft)**2) / (N**2) # normalisation
    freq = fftfreq(len(wave), T/N)
    posval = freq>=0
    freq = freq[posval]
    power = power[posval] # taking positive values only
    return power, freq

def wavegen(t, f, a, p, noise):
    '''
    Calculates signal based on input from signal choice button(s)
    '''
    if wavetype=='Sine':
        return a*np.sin(2*np.pi*f*t + p) + noise
    elif wavetype=='Square':
        return a*signal.square(2*np.pi*f*t + p) + noise
    elif wavetype=='Sawtooth':
        return a*signal.sawtooth(2*np.pi*f*t + p) + noise
    

def updateclose(event):
    '''
    Update function for closing figure window
    '''
    plt.close('all')

def update(val):
    '''
    Primary update function called when any widget is changed
    '''
    # defining frequency/time windowing lines globally as called before definition
    global stplot 
    global etplot 
    global sfplot 
    global efplot

    # removing previous frequency/time windowing lines
    if stplot:
        stplot.remove()
    if etplot:
        etplot.remove()
    if sfplot:
        sfplot.remove()
    if efplot:
        efplot.remove() 
    
    # updating key values based on slider value
    f = fslider.val
    st = stslider.val
    et = etslider.val
    sf = sfslider.val
    ef = efslider.val
    T = tslider.val
    N = Nslider.val * T # ensures number of sampling points is correct based on wave duration
    a = aslider.val
    p = pslider.val * np.pi # puts phase in units of pi
    
    # rescales time window 
    if et-st>T:
        et=st+T
    elif et<st:
        et=st+T
    
    # defining time arrays
    tfull = np.arange(0, T, T/N)
    t = np.arange(st, et, T/N)
    
    # defining noise arrays
    noise = np.random.normal(0, noiseslider.val, len(t))
    noisefull = np.random.normal(0, noiseslider.val, len(tfull))
    
    # calculating signal arrays
    yfull = wavegen(tfull, f, a, p, noisefull)
    y = wavegen(t, f, a, p, noise)
    
    # calculating power spectrum and its frequency values
    power, freq = powspec(yfull, N, T)
    
    # calculating inverse fourier transform
    iftfreq = np.abs(fftfreq(len(y), T/N))
    
    # accounting for frequency windowing
    ft = fft(y)
    ft[iftfreq<sf]=0
    ft[iftfreq>ef]=0  
   
    # updates and plots new data from widget update
    axesHandle.set_ydata(y)
    axesHandle.set_xdata(t)
    axesHandle2.set_ydata(power) 
    axesHandle2.set_xdata(freq)
    axesHandle3.set_ydata(ifft(ft))
    axesHandle3.set_xdata(t)
    
    # plots time and frequency window limit lines
    stplot = ax1.axvline(st, color='deeppink', linestyle='--') # initial time
    etplot = ax1.axvline(et, color='deeppink', linestyle='--') # final time
    sfplot = ax2.axvline(sf, color='deeppink', linestyle='--') # initial frequency
    efplot = ax2.axvline(ef, color='deeppink', linestyle='--') # final frequency
    
    # rescales axis and draws new plots
    ax2.relim()
    ax2.autoscale_view()
    plt.draw()
    
    
def updatesignal(label):
    '''
    Update function for when signal type is changed
    '''
    global wavetype
    wavetype = label
    update(None) # calls upon update function as parameters must change due to signal type change



# defining subplots
fig = plt.figure(figsize=[12,6])
plt.subplots_adjust(wspace=0.3, hspace=0.3, top=0.85, bottom=0.5)

# defining widgets and their axis
closeax = plt.axes([0.85, 0.05, 0.1, 0.1]) # close button
closebutton = widgets.Button(closeax, 'Close', color='lavenderblush')

signalax = plt.axes([0.6, 0.05, 0.12, 0.15]) # signal choice button(s)
signalbutton = widgets.RadioButtons(signalax, ('Sine', 'Square', 'Sawtooth'))

fsax = plt.axes([0.1, 0.35, 0.3, 0.03]) # frequency slider
fslider = widgets.Slider(fsax, 'Frequency [Hz]', 0, 50, valinit=10, color='hotpink')

Nsax = plt.axes([0.1, 0.30, 0.3, 0.03]) # no. of sampling points slider
Nslider = widgets.Slider(Nsax, 'N', 10, 100, valinit=100, color='hotpink', valstep=1)

asax = plt.axes([0.1, 0.25, 0.3, 0.03]) # amplitude slider
aslider = widgets.Slider(asax, 'Amplitude', 0, 2, valinit=1, color='hotpink')

psax = plt.axes([0.1, 0.20, 0.3, 0.03]) # phase slider
pslider = widgets.Slider(psax, 'Phase [pi]', 0, 2, valinit=0, color='hotpink')

tax = plt.axes([0.1, 0.15, 0.3, 0.03]) # time slider
tslider = widgets.Slider(tax, 't [s]', 0.01, 1, valinit=1, color='hotpink')

stax = plt.axes([0.1, 0.1, 0.3, 0.03]) # initial time window slider
stslider = widgets.Slider(stax, 'Initial t [s]', 0, 1, valinit=0, color='hotpink')

etax = plt.axes([0.1, 0.05, 0.3, 0.03]) # final time window slider
etslider = widgets.Slider(etax, 'Final t [s]', 0, 1, valinit=1, color='hotpink')

noiseax = plt.axes([0.55, 0.35, 0.3, 0.03]) # noise level slider
noiseslider = widgets.Slider(noiseax, 'Noise', 0, 1, valinit=0, color='hotpink')

sfax = plt.axes([0.55, 0.30, 0.3, 0.03]) # inital frequency window slider
sfslider = widgets.Slider(sfax, 'Initial f [Hz]', 0, 50, valinit=0, color='hotpink')

efax = plt.axes([0.55, 0.25, 0.3, 0.03]) # final frequency window slider
efslider = widgets.Slider(efax, 'Final f [Hz]', 0, 50, valinit=50, color='hotpink')


## defining initial conditions
N = 100 # no. of sampling points
a = 1 # amplitude
f = 10 # frequency
p = 0 # phase
noise = 0 # noise level
stplot = None
etplot = None
sfplot = None
efplot = None # turning off windowing lines initially
t = np.arange(0, 1, 1/N) 
wavetype = 'Sine'

# calculating initial power spectrum and inverse fourier transform plots
ps, freq = powspec(wavegen(t, f, a, p, noise), N, 1)
inft = ifft(fft(wavegen(t, f, a, p, noise)))


## plotting graphs
ax1 = fig.add_subplot(1, 3, 1)
ax1.set_title('Signal')
ax1.set_xlabel('t [s]')
ax1.set_ylabel('Amplitude')
axesHandle, = ax1.plot(t, wavegen(t, f, a, p, noise), color='pink')

ax2 = fig.add_subplot(1, 3, 2)
ax2.set_title('Power Spectrum')
ax2.set_xlabel('Frequency [Hz]')
ax2.set_ylabel('Amplitude$^2$')
axesHandle2, = ax2.plot(freq, ps, color='mediumvioletred')

ax3 = fig.add_subplot(1, 3, 3)
ax3.set_title('Inverse Fourier Transform')
ax3.set_xlabel('t [s]')
ax3.set_ylabel('Amplitude')
axesHandle3, = ax3.plot(t, inft, color='deeppink')


## calling functions on widget updates
closebutton.on_clicked(updateclose)
signalbutton.on_clicked(updatesignal)
fslider.on_changed(update)
Nslider.on_changed(update)
etslider.on_changed(update)
stslider.on_changed(update)
tslider.on_changed(update)
noiseslider.on_changed(update)
aslider.on_changed(update)
pslider.on_changed(update)
sfslider.on_changed(update)
efslider.on_changed(update)

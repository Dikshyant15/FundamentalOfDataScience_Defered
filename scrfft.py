import numpy as np

def scrfft(xdata,ydata):
    sdata=np.argsort(xdata)
    xdatas=xdata[sdata]
    ydatas=ydata[sdata]
    
    
    xmin=np.min(xdata)
    xmax=np.max(xdata)
    ndata=len(xdata)
    x=(xmax-xmin)/(ndata-1)*np.arange(ndata)+xmin
    y=np.interp(x,xdatas,ydatas)
    
    yf = 2.0*np.fft.rfft(y)/(ndata+1)
    a=np.real(yf)
    b=-np.imag(yf)
    yf=0
    
    a[0]=0.5*a[0]
    
    f=np.arange(len(a))/(xmax-xmin)
    
    return(f, a, b)







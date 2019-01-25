# File: pf.py
# Functions for gnuradio-companion PAM p(t) generation

import numpy as np

def pampt(sps, ptype, pparms=[]):
    """
    PAM pulse p(t) = p(n*TB/sps) generation
    >>>>> pt = pampt(sps, ptype, pparms) <<<<<
    where  sps:
           ptype: pulse type ('rect', 'tri', 'rcf')
           pparms not used for 'rect', 'tri'
           pparms = [k, beta] for 'rcf'
           k: "tail" truncation parameter for 'sinc' (truncates p(t) to -k*sps <= n < k*sps)
           beta: roll-off factor
           pt: pulse p(t) at t=n*TB/sps
    Note: In terms of sampling rate Fs and baud rate FB,
          sps = Fs/FB
    """

    if ptype.lower() == 'rect':
        nn = np.arange(0,sps)
        pt = np.ones(len(nn))

    elif ptype.lower() == 'tri':
        nn = np.arange(-sps, sps)
        pt = np.zeros(len(nn))
        ix = np.where(nn < 0)[0]
        pt[ix] = 1 + nn[ix]/float(sps)
        ix = np.where(nn >= 0)[0]
        pt[ix] = 1 - nn[ix]/float(sps)

    elif ptype == 'rcf':
        nk = round(pparms[0]*sps)
        nn = np.arange(-nk,nk)
        pt = np.sinc(nn/float(sps))
        if len(pparms) > 1:
            p2t = 0.25*np.pi*np.ones(len(nn))
            atFB = pparms[1]/float(sps)*nn
            atFB2 = np.power(2*atFB, 2.0)
            ix = np.where(atFB2 != 1)[0]
            p2t[ix] = np.cos(np.pi*atFB[ix])
            p2t[ix] = p2t[ix]/(1-atFB2[ix])
            pt = pt*p2t

    else:
        pt = np.ones(1) #default value

    return pt

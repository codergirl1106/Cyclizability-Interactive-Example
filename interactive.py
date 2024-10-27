import streamlit as st

import matplotlib.pyplot as plt
import numpy as np
import random
import math
import io
import time
from tensorflow import keras

from scipy.interpolate import interp1d
from scipy.optimize import fsolve

import re
import py3Dmol
from stmol import showmol
import requests
import streamlit_scrollable_textbox as stx
import streamlit_js_eval
import functools
from itertools import chain
import string

def my_cache(f):
    @st.cache_data(max_entries=5, ttl=600)
    @functools.wraps(f)
    def inner(*args, **kwargs):
        return f(*args, **kwargs)
    return inner

@st.cache_data
def load_model(modelnum: int):
    return keras.models.load_model(f"./adapter-free-Model/C{modelnum}free")

def pred(model, pool):
    input = np.zeros((len(pool), 200), dtype = np.single)
    temp = {'A':0, 'T':1, 'G':2, 'C':3}
    for i in range(len(pool)): 
        for j in range(50):
            input[i][j*4 + temp[pool[i][j]]] = 1
    A = model.predict(input, batch_size=128)
    A.resize((len(pool),))
    return A

def getTexttt(pbdid): 
    link = f"https://files.rcsb.org/download/{pbdid}.pdb"
    texttt = requests.get(link).text
    return texttt

def getSequence(pbdid):
    sequencelink = f"https://www.rcsb.org/fasta/entry/{pbdid}"
    tt = requests.get(sequencelink).text
    seq_and_chains = re.findall(f">{pbdid.upper()}_\d|Chains? ([A-Z])([^|]*).*\n([A-Z]+)\n",tt)
    seq_and_chains.sort()
    
    otherlink = f'https://files.rcsb.org/download/{pbdid}.cif'
    tt = requests.get(otherlink).text
    sequences = []
    cords = []
    
    qqq = dict()
    sqq = dict()
    for i in seq_and_chains:
        if i[0] == '' or i[2] == '':
            continue

        im = i[1].split(',')
        chains, seq = [i[0]], i[2]
        if len(im) >= 2:
            chains += list(map(lambda x: x.strip()[0], im[1:]))
        
        stuff = re.findall(f'ATOM[^\S\r\n]+(\d+)[^\S\r\n]+([A-Z])[^\S\r\n]+(\"?[A-Z]+\d*\'?\"?)[^\S\r\n]+(.)[^\S\r\n]+([A-Z]+)[^\S\r\n]+([{"|".join(chains)}])[^\S\r\n]+([0-9]+)[^\S\r\n]+([0-9]+)[^\S\r\n]+\?[^\S\r\n]+'+"(-?\d+[\.\d]*)[^\S\r\n]*"*5+'.+\n', tt)
        if set(seq[int(stuff[0][7])-1:int(stuff[-1][7])]).issubset({'A', 'C', 'G', 'T'}):
            # find helical axis
            for line in stuff:
                if line[4] in ['DA', 'DG'] and line[2] == 'C8':
                    sqq[line[5]] = seq[int(stuff[0][7])-1:int(stuff[-1][7])]
                    if line[5] in qqq:
                        qqq[line[5]].append([float(line[_]) for _ in range(8, 11)])
                    else:
                        qqq.update({line[5]: [[float(line[_]) for _ in range(8, 11)]]})
                    
                if line[4] in ['DT', 'DC'] and line[2] == 'C6':
                    sqq[line[5]] = seq[int(stuff[0][7])-1:int(stuff[-1][7])]
                    if line[5] in qqq:
                        qqq[line[5]].append([float(line[_]) for _ in range(8, 11)])
                    else:
                        qqq.update({line[5]: [[float(line[_]) for _ in range(8, 11)]]})

    alphabet = string.ascii_uppercase
    if len(qqq) % 2 == 0:
        while True:
            keyd = sorted(qqq.keys())
            if len(keyd) == 0:
                break
            keyn = alphabet[alphabet.index(keyd[0])+1]
            sequences.append(sqq[keyd[0]])
            qqq[keyd[0]] = qqq[keyd[0]]+qqq[keyn]
            del qqq[keyn]
            qwer, asdf = np.array(qqq[keyd[0]][:len(qqq[keyd[0]])//2], dtype = np.single), np.array(qqq[keyd[0]][-(len(qqq[keyd[0]])//2):][::-1], dtype = np.single)
            del qqq[keyd[0]]
            otemp = (qwer+asdf)/2
            o = np.array([(otemp[i+1]+otemp[i])/2 for i in range(len(otemp)-1)], dtype = np.single)
            ytemp = qwer - asdf
            z = np.array([otemp[i+1]-otemp[i] for i in range(len(otemp)-1)], dtype = np.single)
            ytemp = [(ytemp[i+1]+ytemp[i])/2 for i in range(len(ytemp)-1)]
            x = np.array([np.cross(ytemp[i], z[i]) for i in range(len(z))], dtype = np.single)
            y = np.array([np.cross(z[i], x[i]) for i in range(len(z))], dtype = np.single)
            x = -np.array([x[i]/np.linalg.norm(x[i]) for i in range(len(x))], dtype = np.single) # direction of minor groove
            y = np.array([y[i]/np.linalg.norm(y[i]) for i in range(len(y))], dtype = np.single)
            cords.append((o, x, y))
    elif len(qqq) == 3:
        #print("hullo")
        main_seq = sorted(sqq.items(), key=lambda x: len(x[1]))[-1][0]
        #print(main_seq)
        for i in qqq:
            if i != main_seq:
                sequences.append(sqq[i])
                #qqq[i] = qqq[i]+qqq[main_seq]
                qwer, asdf = np.array(qqq[i][:len(qqq[i])//2], dtype = np.single), np.array(qqq[i][-(len(qqq[i])//2):][::-1], dtype = np.single)
                otemp = (qwer+asdf)/2
                o = np.array([(otemp[i+1]+otemp[i])/2 for i in range(len(otemp)-1)], dtype = np.single)
                ytemp = qwer - asdf
                z = np.array([otemp[i+1]-otemp[i] for i in range(len(otemp)-1)], dtype = np.single)
                ytemp = [(ytemp[i+1]+ytemp[i])/2 for i in range(len(ytemp)-1)]
                x = np.array([np.cross(ytemp[i], z[i]) for i in range(len(z))], dtype = np.single)
                y = np.array([np.cross(z[i], x[i]) for i in range(len(z))], dtype = np.single)
                x = -np.array([x[i]/np.linalg.norm(x[i]) for i in range(len(x))], dtype = np.single) # direction of minor groove
                y = np.array([y[i]/np.linalg.norm(y[i]) for i in range(len(y))], dtype = np.single)
                cords.append((o, x, y))
    else:
        for i in qqq:
            sequences.append(sqq[i])
            qwer, asdf = np.array(qqq[i][:len(qqq[i])//2], dtype = np.single), np.array(qqq[i][-(len(qqq[i])//2):][::-1], dtype = np.single)
            otemp = (qwer+asdf)/2
            o = np.array([(otemp[i+1]+otemp[i])/2 for i in range(len(otemp)-1)], dtype = np.single)
            ytemp = qwer - asdf
            z = np.array([otemp[i+1]-otemp[i] for i in range(len(otemp)-1)], dtype = np.single)
            ytemp = [(ytemp[i+1]+ytemp[i])/2 for i in range(len(ytemp)-1)]
            x = np.array([np.cross(ytemp[i], z[i]) for i in range(len(z))], dtype = np.single)
            y = np.array([np.cross(z[i], x[i]) for i in range(len(z))], dtype = np.single)
            x = -np.array([x[i]/np.linalg.norm(x[i]) for i in range(len(x))], dtype = np.single) # direction of minor groove
            y = np.array([y[i]/np.linalg.norm(y[i]) for i in range(len(y))], dtype = np.single)
            cords.append((o, x, y))
    
    return sequences, cords

def envelope(fity):
    ux, uy = [0], [fity[0]]
    lx, ly = [0], [fity[0]]
    
    # local extremas
    for i in range(1, len(fity)-1):
        if (fity[i] == max(fity[max(0, i-3):min(i+4, len(fity))])):
            ux.append(i)
            uy.append(fity[i])
        if (fity[i] == min(fity[max(0, i-3):min(i+4, len(fity))])):
            lx.append(i)
            ly.append(fity[i])

    ux.append(len(fity)-1)
    uy.append(fity[-1])
    lx.append(len(fity)-1)
    ly.append(fity[-1])
    
    ub = np.array([fity, interp1d(ux, uy, kind=3, bounds_error=False)(range(len(fity)))]).max(axis=0)
    lb = np.array([fity, interp1d(lx, ly, kind=3, bounds_error=False)(range(len(fity)))]).min(axis=0)
    return ub-lb

def trig(x, *args): # x = [C0, amp, psi]
    return [args[0][0] - x[0] - x[1]**2*math.cos((34.5/args[0][-1]-3)*2*math.pi-math.pi*2/3 - x[2]),
            args[0][1] - x[0] - x[1]**2*math.cos((31.5/args[0][-1]-3)*2*math.pi-math.pi*2/3 - x[2]),
            args[0][2] - x[0] - x[1]**2*math.cos((29.5/args[0][-1]-2)*2*math.pi-math.pi*2/3 - x[2])]

def show_st_3dmol(pdb_code,original_pdb,cartoon_style="oval",
                  cartoon_radius=0.2,cartoon_color="lightgray",zoom=1,spin_on=True):
    
    view = py3Dmol.view(width=300, height=300)
        
    view.addModelsAsFrames(pdb_code)
    view.addModelsAsFrames(original_pdb)
    view.setStyle({"cartoon": {"style": cartoon_style,"color": cartoon_color,"thickness": cartoon_radius}})

    style_lst = []
    surface_lst = []
    
    view.addStyle({'chain':'Z'}, {'stick': {"color": "blue"}})

    view.addStyle({'chain':'A'}, {'line': {}})
    view.addStyle({'chain':'B'}, {'line': {}})
    view.addStyle({'chain':'C'}, {'line': {}})
    view.addStyle({'chain':'D'}, {'line': {}})
    view.addStyle({'chain':'E'}, {'line': {}})
    view.addStyle({'chain':'F'}, {'line': {}})
    view.addStyle({'chain':'G'}, {'line': {}})
    view.addStyle({'chain':'H'}, {'line': {}})
    view.addStyle({'chain':'I'}, {'line': {}})
    view.addStyle({'chain':'J'}, {'line': {}})
    
    view.zoomTo()
    view.spin(spin_on)
    view.zoom(zoom)

    showmol(view, height=300, width=300)
    return 0

def helpercode(model_num: int, seqlist: dict, pool, sequence):
    prediction = pred(load_model(model_num), tuple(seqlist.keys()))

    result_array = np.zeros((len(pool), len(pool[0]) - 49), dtype = np.single)

    for i in range(len(pool)):
        for j in range(49):
            result_array[i, j] = prediction[seqlist[pool[i][j:j + 50]]]
        for j in range(len(pool[i])-98, len(pool[i])-49):
            result_array[i, j] = prediction[seqlist[pool[i][j:j + 50]]]
    
    result_array = result_array.mean(axis=0)
    if len(sequence) >= 50:
        seqlist = [sequence[i:i+50] for i in range(len(sequence)-49)]
        result_array[49:-49] = pred(load_model(model_num), seqlist).reshape(-1, )

    result_array -= result_array.mean()
    return result_array

def pdb_out(name, psi, amp, factor, cords):
    counterhetatm = 0
    counterconect = 0

    pdb_final_output = "HEADER    output from spatial analysis\n"
    hetatmf = ''
    conectf = ''
    for i in range(len(cords)):
        arrow = []
        
        for j in range(len(cords[i][1])):
            arrow.append(np.cos(-psi[i][j])*cords[i][1][j] + np.sin(-psi[i][j])*cords[i][2][j])
        
        arrow = np.array(arrow)

        for j in range(len(arrow)):
            arrow[j] /= np.linalg.norm(arrow[j])
            arrow[j] *= factor*amp[i][j]
        
        e = cords[i][0] + arrow  # OH THIS ERROR OCCURS BECAUSE THERE ARE TWO SEQUENCES. FIGURE OUT HOW TO INCORPERATE THE SECOND SEQUENCE.
        o = np.around(cords[i][0], 3)
        e = np.around(e, 3)

        hetatm = ''
        conect = ''
        for j in range(len(o)):
            hetatm += 'HETATM' + str(counterhetatm+j+1).rjust(5) + ' C    AXI ' + 'Z' + '   1    ' # '    1' 5d
            for k in range(3):
                hetatm += str(o[j][k]).rjust(8)
            hetatm += '\n'
        for j in range(len(e)):
            hetatm += 'HETATM' + str(counterhetatm+j+1+len(o)).rjust(5) + ' C    AXI ' + 'Z' + '   1    ' # '    1' 5d
            for k in range(3):
                hetatm += str(round(e[j][k], 2)).rjust(8)
            hetatm += '\n'
        for j in range(len(o)-1):
            conect += 'CONECT' + str(counterconect+j+1).rjust(5) + str(counterconect+j+2).rjust(5) + '\n' # CONECT    1    2
        for j in range(len(o)):
            conect += 'CONECT' + str(counterconect+j+1).rjust(5) + str(counterconect+j+1+len(o)).rjust(5) + '\n' # CONECT    1    2

        counterhetatm += 2*len(o)
        counterconect += 2*len(o)
        hetatmf += hetatm
        conectf += conect

    return pdb_final_output + hetatmf + conectf

@my_cache
def longcode(sequence, helical_turn):
    pool = []
    base = ['A','T','G','C']

    for i in range(200):
        left = ''.join([random.choice(base) for i in range(49)])
        right = ''.join([random.choice(base) for i in range(49)])
        pool.append(left + sequence + right)

    seqlist = dict()
    indext = 0
    for i in range(len(pool)):
        for j in range(len(pool[i])-49):
            tt = pool[i][j:j+50]
            if tt not in seqlist:
                seqlist.update({tt: indext})
                indext += 1
    
    models = dict.fromkeys((26, 29, 31))
    for modelnum in models.keys():
        models[modelnum] = helpercode(modelnum, seqlist, pool, sequence)
        
    amp = sum(envelope(m) for m in models.values()) / len(models)
    
    psi = []
    for i in range(len(amp)):
        root = fsolve(trig, [1, 1, 1], args=[m[i] for m in models.values()]+[helical_turn])
        psi.append(root[2])
        if(psi[-1] > math.pi): psi[-1] -= 2*math.pi

    psi = np.array(psi, dtype = np.single)

    # trim random sequences
    amp,psi = amp[25:-25],psi[25:-25]
    return amp, psi

def spatial_analysis_ui(imgg, sequence, texttt, cords):
        
    factor = 30

    helical_turn = 10.3
    
    amp, psi = [], []
    for seq in sequence:
        a, p = longcode(seq, helical_turn)
        amp.append(a)
        psi.append(p)
    
    pdb_output = pdb_out(texttt, psi, amp, factor, cords)

    
    show_st_3dmol(pdb_output,texttt)
    
def main():
     st.markdown(
        """
        <style>
        .reportview-container {
            background: #FFFFFF
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    texttt = getTexttt("7OHC")
    seq, cords = getSequence("7OHC")
    imgg = io.BytesIO()
    spatial_analysis_ui(imgg, seq, texttt, cords)

main()

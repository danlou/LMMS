import numpy
import cPickle
import scipy
import scipy.stats
import pdb
import os
import theano
import theano.tensor as T
import sys
from dataset.DatasetInterfaces import root

from util.embedding import knn, display
from util.cost import nll, hardmax, cepond, nllsoft,ce
from util.expr import rect, identity, hardtanh
from util.io import save, load
from util.sparse import idx2spmat, idx2mat, idx2vec

from unsupervised import cae,ae
from supervised import logistic
from jobscheduler import JobmanInterface as JB
from tools.hyperparams import hyperparams

import sparse.supervised

from theano import function

def parse_data(f='ratings.txt', vocab='/scratch/rifaisal/data/wiki_april_2010/WestburyLab.wikicorp.201004_vocab30k.pkl'):
    vocab = cPickle.load(open(vocab))
    print len(vocab)
    dvocab = dict(zip(vocab,range(len(vocab))))
    def map2vocab(s):
        cidx = []
        for w in s:
            try:
                cidx.append(dvocab[w])
            except:
                #print 'Word not found:',w
                cidx.append(dvocab['UUUKKKNNN'])
        return cidx


    docs = open(f,'r')
    swcs = []

    for i,d in enumerate(docs):
        split1 = d.lower().split('\t')
        idx = int(split1[0])
        w1 = split1[1]
        w2 = split1[3]
        scores = [ float(dd) for dd in split1[-11:] ][1:]
        #print len(scores)
        #print scores
        c1,c2 = split1[5:-11]

        c1 = [ w.replace('<b>','') for w in c1.split(' ') ]
        c2 = [ w.replace('<b>','') for w in c2.split(' ') ]

        w1idx = [ j for j,w in enumerate(c1[:-1]) if w1 == w and c1[j+1] == '</b>' ]
        w2idx = [ j for j,w in enumerate(c2[:-1]) if w2 == w and c2[j+1] == '</b>' ]

        c1 = [ w for w in c1 if w != '</b>' ]
        c2 = [ w for w in c2 if w != '</b>' ]

        assert len(w1idx) == 1
        assert len(w2idx) == 1
        c1idx = map2vocab(c1)
        c2idx = map2vocab(c2)
        element = (scores,w1idx,w2idx,c1idx,c2idx)
        swcs.append(element)
    return swcs

def score(jobman,path):
    hp = jobman.state
    nsenna = 30000

    PATH = '/scratch/rifaisal/msrtest/test/'
    delta = hp['wsize']/2
    rest = hp['wsize']%2
    sent = T.matrix()

    embedding = cae(i_size=nsenna, h_size=hp['embedsize'], e_act = identity)
    H = ae(i_size = hp['embedsize']*hp['wsize'], h_size=hp['hsize'], e_act = T.tanh)
    L = logistic(i_size = hp['hsize'], h_size = 1, act = identity)

    load(embedding,path+'/embedding.pkl')
    load(H,path+'/hidden.pkl')
    load(L,path+'/logistic.pkl')

    posit_embed = T.dot(sent, embedding.params['e_weights']).reshape((1,hp['embedsize']*hp['wsize']))
    posit_score = H.encode(posit_embed)
    scoreit = theano.function([sent],posit_score)
    sentences = parse_data()
    scores = []
    esims = []
    msim = []
    hsim = []
    Em = embedding.params['e_weights'].get_value(borrow=True)
    for i,(sc,w1,w2,c1,c2) in enumerate(sentences):
        sys.stdout.flush()


        c1 = [29999]*10 + c1 + [29999]*10
        c2 = [29999]*10 + c2 + [29999]*10
        
        w1seqs = [ c1[10+idx-delta:10+idx+delta+rest] for idx in w1 ]
        w2seqs = [ c2[10+idx-delta:10+idx+delta+rest] for idx in w2 ]

        c =[]

        

        w1em = Em[c1[10+w1[0]]]
        w2em = Em[c2[10+w2[0]]]

        w1sc = numpy.concatenate([ scoreit( idx2mat(w1seqs[0],nsenna) ).flatten() , Em[c1[10+w1[0]]] ])
        w2sc = numpy.concatenate([ scoreit( idx2mat(w2seqs[0],nsenna) ).flatten() , Em[c2[10+w2[0]]] ])

        metric = L.params['weights'].get_value(borrow=True).flatten()

        sim = -(((w1sc - w2sc))**2).sum()
        esim = -((w1em - w2em)**2).sum()

        msim.append(sim)
        esims.append(esim)
        hsim.append(numpy.mean(sc))
                           
    print 'Model:',scipy.stats.spearmanr(numpy.array(hsim), numpy.array(msim))[0],', Embeddings:',scipy.stats.spearmanr(numpy.array(hsim), numpy.array(esims))[0]
    
def jobman_entrypoint(state, channel):
    jobhandler = JB.JobHandler(state,channel)
    for i in range(1,13):
        path = state['loadpath']+'/'+str(i)+'/files'
        print '---------Computing:',i,
        score(jobhandler,path)
    return 


if __name__ == "__main__":
    HP_init = [ ('values','epoch',[100]),
                ('values','deviation',[.1]),
                ('values','iresume',[3756]),
                ('values','freq',[10000]),
                ('values','loadpath',['/scratch/rifaisal/exp/mullerx_db/wikibaseline_bugfix_0002/']),
                ('values','hsize',[100]),
                ('values','embedsize',[50]),
                ('values','wsize',[9]),
                ('values','npos',[1]),
                ('values','nneg',[1,10]),
                ('values','lr',[0.01,.1]),
                ('values','lambda',[.0]),
                ('values','margin',[1.]),
                ('values','bs',[10]) ]

    hp_sampler = hyperparams(HP_init)
    jobparser = JB.JobParser(jobman_entrypoint,hp_sampler)


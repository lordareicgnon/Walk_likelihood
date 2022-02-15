import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi
def walk_likelihood(X,Ws,ket,lm=7,max_iter=20,thr=0.001):
    ketpr=ket.copy()
    N=len(ket)
    m2=len(ket[0,:])
    converged=0
    amx_pr=np.argmax(ket,axis=1)
    for ai in range(max_iter):
        dketb=X.dot(ket)
        ketb=dketb
        for i in range(lm):
            dketb=X.dot(dketb/Ws[:,None])
            ketb=ketb+dketb
        Ks=np.dot(np.transpose(ketb),ket)
        smk=(np.diagonal(Ks)>0)
        if (np.prod(smk)==0):
            ket=ket[:,smk]
            ketpr=ketpr[:,smk]
            ketb=ketb[:,smk]
            Ks=Ks[smk,:][:,smk]
            m2=len(ket[0,:])
        wtots=np.dot(np.transpose(Ws),ket)
        score=(Ks/wtots)
        fac=1/np.diagonal(score)
        #fac=1/np.max(score,axis=1)
        lg_score=((np.log(score+0.00000001*(score==0)/wtots)).astype('float'))*fac[:,None]
        score=score*fac[:,None]
        amxx=np.argmax(np.dot(ketb,lg_score)-np.outer(Ws,sum((score))),axis=1)
        ket=np.zeros_like(np.zeros((N,m2)))
        ket[range(len(amxx)),amxx]=1
        #samm=sum(((ket>0)*(ketpr>0)))/np.maximum(sum(ketpr>0),sum(ket>0))#*Ws[:,None])/sum(ketpr*Ws[:,None])
        #converged=sum((1-samm)<thr)==m2
        converged=nmi(amx_pr,amxx)>0.99
        converged=(1-(np.sum(ket*ketpr)/N))<thr
        if converged:
            break
        ketpr=ket.copy()
        amx_pr=amxx.copy()
    smk=sum(ket)>0
    if (np.prod(smk)==0):
        ket=ket[:,smk]
    return ket

def walk_likelihood_alt(X,Ws,ket,lm=7,max_iter=100,thr=0.001):
    ketpr=ket.copy()
    N=len(ket)
    m2=len(ket[0,:])
    converged=0

    for ai in range(max_iter):
        dketb=X.dot(ket)
        ketb=dketb
        for i in range(lm):
            dketb=X.dot(dketb/Ws[:,None])
            ketb=ketb+dketb
        Ks=np.dot(np.transpose(ketb),ket)
        smk=(np.diagonal(Ks)>0)
        if (np.prod(smk)==0):
            ket=ket[:,smk]
            ketpr=ketpr[:,smk]
            ketb=ketb[:,smk]
            Ks=Ks[smk,:][:,smk]
            m2=len(ket[0,:])
        wtots=np.dot(np.transpose(Ws),ket)
        score=(Ks/wtots)
        fac=1/np.diagonal(score)
        #fac=1/np.max(score,axis=1)
        lg_score=((np.log(score+0.00000001*(score==0)/wtots)).astype('float'))*fac[:,None]
        score=score*fac[:,None]
        amxx=np.argmax(np.dot(ketb,lg_score)-np.outer(Ws,sum((score))),axis=1)
        ket=np.zeros_like(np.zeros((N,m2)))
        ket[range(len(amxx)),amxx]=1

        ketb=X.dot(ket)
        Ks=np.dot(np.transpose(ketb),ket)
        smk=(np.diagonal(Ks)>0)
        if (np.prod(smk)==0):
            ket=ket[:,smk]
            ketpr=ketpr[:,smk]
            ketb=ketb[:,smk]
            Ks=Ks[smk,:][:,smk]
            m2=len(ket[0,:])
        wtots=np.dot(np.transpose(Ws),ket)
        score=(Ks/wtots)
        fac=1/np.diagonal(score)
        #fac=1/np.max(score,axis=1)
        lg_score=((np.log(score+0.00000001*(score==0)/wtots)).astype('float'))*fac[:,None]
        score=score*fac[:,None]
        amxx=np.argmax(np.dot(ketb,lg_score)-np.outer(Ws,sum((score))),axis=1)
        ket=np.zeros_like(np.zeros((N,m2)))
        ket[range(len(amxx)),amxx]=1
        #samm=sum(((ket>0)*(ketpr>0)))/np.maximum(sum(ketpr>0),sum(ket>0))#*Ws[:,None])/sum(ketpr*Ws[:,None])
        #converged=sum((1-samm)<thr)==m2
        converged=(1-(np.sum(ket*ketpr)/N))<thr

        if converged:
            break
        ketpr=ket.copy()
    smk=sum(ket)>0
    if (np.prod(smk)==0):
        ket=ket[:,smk]
    return ket

def walk_likelihood_overlapping(X,Ws,ket,lm=7,max_iter=20,thr=0.001):
    ketpr=ket.copy()
    N=len(ket)
    m2=len(ket[0,:])
    converged=0
    keta=ket.copy()
    for ai in range(max_iter):
        #print(ai)
        dketb=X.dot(ket)
        ketb=dketb
        for i in range(lm):
            dketb=X.dot(dketb/Ws[:,None])
            ketb=ketb+dketb
        Ks=np.dot(np.transpose(ketb),ket)
        smk=(np.diagonal(Ks)>0)
        if (np.prod(smk)==0):
            ket=ket[:,smk]
            ketpr=ketpr[:,smk]
            ketb=ketb[:,smk]
            Ks=Ks[smk,:][:,smk]
            m2=len(ket[0,:])
        wtots=np.dot(np.transpose(Ws),ket)
        score=(Ks/wtots)
        fac=1/np.diagonal(score)
        #fac=1/np.max(score,axis=1)
        lg_score=((np.log(score+0.00000001*(score==0)/wtots)).astype('float'))*fac[:,None]
        score=score*fac[:,None]
        amxx=np.argmax(np.dot(ketb,lg_score)-np.outer(Ws,sum((score))),axis=1)
        keta=np.zeros_like(np.zeros((N,m2)))
        keta[range(len(amxx)),amxx]=1
        #samm=sum(((ket>0)*(ketpr>0)))/np.maximum(sum(ketpr>0),sum(ket>0))#*Ws[:,None])/sum(ketpr*Ws[:,None])
        #converged=sum((1-samm)<thr)==m2
        Qa=abs((score-np.diagonal(score)[:,None])/(lg_score-np.diagonal(lg_score)[:,None]))
        Qa[score>1]=1000000
        np.fill_diagonal(Qa,0)
        ket=ketb>(keta.dot((Qa))*Ws[:,None])
        converged=(1-(np.sum(ket*ketpr)/np.sum(ket)))<thr
        if converged:
            break
        ketpr=ket.copy()
    smk=sum(ket)>0
    if (np.prod(smk)==0):
        ket=ket[:,smk]
        keta=keta[:,smk]
    #print(keta)
    return keta


def walk_likelihood3(X,Ws,ket,lm=7,max_iter=20,thr=0.001):
    ketpr=ket.copy()
    N=len(ket)
    m2=len(ket[0,:])
    converged=0
    g=np.ones(m2)
    done=0
    for ai in range(max_iter):
        dketb=X.dot(ket)
        ketb=dketb
        for i in range(lm):
            dketb=X.dot(dketb/Ws[:,None])
            ketb=ketb+dketb
        Ks=np.dot(np.transpose(ketb),ket)
        smk=(np.diagonal(Ks)>0)
        if (np.prod(smk)==0):
            ket=ket[:,smk]
            ketpr=ketpr[:,smk]
            ketb=ketb[:,smk]
            g=g[smk]
            Ks=Ks[smk,:][:,smk]
            m2=len(ket[0,:])
        wtots=np.dot(np.transpose(Ws),ket)
        #Wtot=np.sum(Ks)
        #K2=np.diagonal(Ks)**2/wtots
        #Wtot=np.sum(K2)
        score=(Ks/wtots)
        #score=(Ks/np.sum(Ks))
        #fac=(Wtot/np.diagonal(Ks))**(1/np.diagonal(Ks))/np.diagonal(score)
        #fac=np.exp(np.log(Wtot*K2)/K2)/np.diagonal(score)
        #Wtot=np.sum(Ks)
        #wtts=np.sum(Ks,axis=1)/Wtot
        #fac=(np.diagonal(Ks)/Wtot-wtts**2)/wtots
        #fac=1/np.max(score,axis=1)

        fac=1/np.diagonal(score)#*np.log(np.diagonal(score)))
        #fac=(-np.diagonal(score)+2*np.sum(score,axis=1))/np.diagonal(score)
        inns=1#/np.sum(ket)#wtots#/np.diagonal(score)
        #for tr in range(10):
        #scr2=score*fac[:,None]
        #scr2[scr2==1]=0
        #avg=0.001*np.max(scr2,axis=1)
        #avg=np.sum(scr2,axis=1)/m2
        #fac=fac/(1+(1-avg)/np.log(avg))
            #fac=fac*(1+np.log(np.sum(scr2,axis=1)))
        #fac=fac*np.sum(scr2,axis=1)
        #    fac=fac/np.sum(fac)
        score=score*fac[:,None]
        Ks=Ks*fac[:,None]
        lg_score=(np.log(score+0.00000001*(score==0)/wtots))
        g=np.exp(-(np.sum(score*(lg_score),axis=1))/np.sum(score,axis=1))
        #print(g)
        #(ket)/wtots
        if ai<-1 or done:
            #g=np.ones(m2)
            for i in range(100):
                #inns=1/g
                g_prev=g.copy()
                F=np.transpose(np.exp(0.0000001*(np.dot(Ks*g*inns,lg_score)-np.dot(score,g*wtots*inns))))
                #F=np.transpose(np.exp(0.0000001*(np.dot(score,lg_score)-np.dot(score,g*wtots))))
                #print(F)
                R=F/sum(F)
                #amx=np.argmax(R,axis=0)
                #R[:,:]=0
                #R[amx,range(m2)]=1
                g=(wtots*(np.sum(score.dot(R),axis=1)-np.diagonal(lg_score.dot(R.dot(Ks*inns)))))
                #h=(inns*wtots*(np.sum(score.dot(R),axis=1)-np.diagonal(lg_score.dot(R.dot(Ks*inns)))))
                #g=h*np.diagonal(R)*(1-np.log(np.diagonal(R)))
                #g=(wtots*(np.sum(score.dot(R),axis=1)-np.diagonal(lg_score.dot(R.dot(score)))))
                g=g*m2/np.sqrt(np.sum(g*g))
                if (np.dot(g-g_prev,g-g_prev)<0.01):
                    #print('here')
                    break
        if ai<-1:
            #g=np.ones(m2)
            lgQ=-lg_score
            Q=score
            #C=np.diagonal(lgQ.dot(Ks))+wtots.dot(Q)
            g=m2*g/np.sum(g)
            Dn=np.sum(Q*(1+lgQ),axis=1)
            for i in range(100):
                g_prev=g.copy()
                F=-np.dot(np.transpose(Q)*g,lgQ)-g.dot(Q)#*wtots
                #F=F-np.diagonal(F)[:,None]
                #print(F)
                F=np.exp(0.0000001*F)
                #amx=np.argmax(F,axis=1)
                #print(amx)
                #R=np.zeros((m2,m2))
                #R[range(m2),amx]=1
                R=F/np.sum(F,axis=1)
                Nm=np.sum(Q.dot(R)*lgQ+Q.dot(np.transpose(R)),axis=1)
                g+=(Nm-Dn)
                g=g*(g>0)
                g=m2*g/np.sum(g)
                if (np.dot(g-g_prev,g-g_prev)<0.000001):
                    #print('here')
                    break
        if ai<-1:
            #g=np.ones(m2)
            lgQ=-lg_score
            Q=score
            V=sum(Ks)
            Wtot=sum(V)
            P=V/Wtot
            for i in range(100):
                g_prev=g.copy()
                R=np.exp(0.00001*(-(g*V).dot(lgQ)-Wtot*(g.dot(Q))))
                R=R/sum(R)
                Nm=Wtot*(Q.dot(R*R))+(lgQ.dot(R*R))*V
                Dm=Wtot*Q.dot(P*R)+V*(lgQ.dot(P*R))
                g=Nm/Dm
                g=g*m2/sum(g)
                if (np.dot(g-g_prev,g-g_prev)<0.000001):
                    #print('here')
                    break
        #input(g)

        score=score*g[:,None]
        lg_score=lg_score*(g*fac)[:,None]
        #lg_score=(np.log(score+0.00000001*(score==0)/wtots))*fac[:,None]
        amxx=np.argmax(np.dot(ketb,lg_score)-np.outer(Ws,sum((score))),axis=1)
        ket=np.zeros_like(np.zeros((N,m2)))
        ket[range(len(amxx)),amxx]=1
        ketb=ketb-ketb[range(len(amxx)),amxx][:,None]
        ketb[range(len(amxx)),amxx]=-1000000
        mx=np.max(ketb,axis=1)
        ket=ket*(mx<-0.1)[:,None]
        #samm=sum(((ket>0)*(ketpr>0)))/np.maximum(sum(ketpr>0),sum(ket>0))#*Ws[:,None])/sum(ketpr*Ws[:,None])
        #converged=sum((1-samm)<thr)==m2
        converged=(1-(np.sum(ket*ketpr)/N))<thr
        if converged:
            break
        ketpr=ket.copy()
    return ket
def walk_likelihood2(X,Ws,ket,lm=7,max_iter=100,thr=0.001):
    ketpr=ket.copy()
    N=len(ket)
    m2=len(ket[0,:])
    converged=0
    for ai in range(max_iter):
        dketb=X.dot(ket)
        ketb=dketb
        for i in range(lm):
            dketb=X.dot(dketb/Ws[:,None])
            ketb=ketb+dketb
        Ks=np.dot(np.transpose(ketb),ket)
        smk=(np.diagonal(Ks)>0)
        if (np.prod(smk)==0):
            ket=ket[:,smk]
            ketpr=ketpr[:,smk]
            ketb=ketb[:,smk]
            Ks=Ks[smk,:][:,smk]
            m2=len(ket[0,:])
        wtots=np.dot(np.transpose(Ws),ket)
        score=(Ks/wtots)
        fac=1/np.diagonal(score)
        fac[:]=1
        lg_score=(np.log(score+0.00000001*(score==0)/wtots))*fac[:,None]
        score=score*fac[:,None]
        amxx=np.argmax(np.dot(ketb,lg_score)-np.outer(Ws,sum((score))),axis=1)
        ket=np.zeros_like(np.zeros((N,m2)))
        ket[range(len(amxx)),amxx]=1
        samm=sum(((ket>0)*(ketpr>0)))/np.maximum(sum(ketpr>0),sum(ket>0))#*Ws[:,None])/sum(ketpr*Ws[:,None])
        converged=sum((1-samm)<thr)==m2
        if converged:
            break
        ketpr=ket.copy()
    return ket
def walk_likelihood_maximize_modularity(X,Ws,ket,lm=7,max_iter=20,thr=0.001,twice=0):
    nstp=1
    Wtot=np.sum(Ws)
    #ket=walk_likelihood(X,Ws,ket,lm=lm,max_iter=max_iter,thr=thr)
    while nstp:

        if twice==1:
            ket=walk_likelihood_alt(X,Ws,ket,lm=lm,max_iter=max_iter,thr=thr)
        elif twice==0:
            ket=walk_likelihood(X,Ws,ket,lm=lm,max_iter=max_iter,thr=thr)
        elif twice==2:
            ket=walk_likelihood(X,Ws,ket,lm=lm,max_iter=max_iter,thr=thr)
            ket=walk_likelihood(X,Ws,ket,lm=0,max_iter=max_iter,thr=thr)
        elif twice==3:
            ket=walk_likelihood(X,Ws,ket,lm=0,max_iter=max_iter,thr=thr)
            ket=walk_likelihood(X,Ws,ket,lm=lm,max_iter=max_iter,thr=thr)
            ket=walk_likelihood(X,Ws,ket,lm=0,max_iter=max_iter,thr=thr)
        elif twice==4:
            ket=walk_likelihood(X,Ws,ket,lm=0,max_iter=max_iter,thr=thr)
            ket=walk_likelihood_alt(X,Ws,ket,lm=lm,max_iter=max_iter,thr=thr)
            ket=walk_likelihood(X,Ws,ket,lm=0,max_iter=max_iter,thr=thr)
        m2=len(ket[0,:])
        wtots=np.dot(np.transpose(Ws),ket)
        M=np.dot(np.transpose(ket),X.dot(ket))-np.outer(wtots,wtots)/Wtot
        np.fill_diagonal(M,0)
        if (np.sum(M>0)>0):
            amx=np.argmax(M)
            #input(np.max(M))
            i=int(amx/m2)
            j=amx-i*m2
            ket[:,i]+=ket[:,j]
            ket=ket[:,np.array(list(range(j))+list(range(j+1,m2)))]
        else:
            nstp=0
    #ket=walk_likelihood(X,Ws,ket,lm=lm,max_iter=max_iter,thr=thr)
    return ket
#'''

def converged(ket1,ket2,thr):
    N=len(ket1)
    if len(ket1[0,:])==len(ket2[0,:]):
        return (N-(np.sum(ket1*ket2)))<(N*thr)
    else:
        return 0
from sklearn.decomposition import NMF
from sklearn.utils.extmath import randomized_svd, safe_sparse_dot, squared_norm
def norm(x):
    return np.sqrt(squared_norm(x))
def clusters_matrix(ket,eps=1e-6):
    amx=np.argmax(ket,axis=1)
    ket[:,:]=0
    ket[range(len(amx)),amx]=1
    ket=ket*(ket>eps)
    return ket
def modularity(X,Ws,ket):
    Wtot=np.sum(Ws)
    wtots=np.dot(np.transpose(Ws),ket)
    e_ii=np.sum(ket*X.dot(ket))/Wtot
    a_ii=wtots/Wtot
    return (np.sum(e_ii)-np.sum(a_ii*a_ii))

def nndsvd(X, n_components, Ws, eps=1e-6):
    U, S, V = randomized_svd(X, n_components, random_state=None)
    W = np.zeros_like(U)
    H = np.zeros_like(V)
    W[:, 0] = np.sqrt(S[0]) * np.abs(U[:, 0])
    H[0, :] = np.sqrt(S[0]) * np.abs(V[0, :])

    for j in range(1, n_components):
        x, y = U[:, j], V[j, :]

        # extract positive and negative parts of column vectors
        x_p, y_p = np.maximum(x, 0), np.maximum(y, 0)
        x_n, y_n = np.abs(np.minimum(x, 0)), np.abs(np.minimum(y, 0))

        # and their norms
        x_p_nrm, y_p_nrm = norm(x_p), norm(y_p)
        x_n_nrm, y_n_nrm = norm(x_n), norm(y_n)

        m_p, m_n = x_p_nrm * y_p_nrm, x_n_nrm * y_n_nrm

        # choose update
        if m_p > m_n:
            u = x_p / x_p_nrm
            v = y_p / y_p_nrm
            sigma = m_p
        else:
            u = x_n / x_n_nrm
            v = y_n / y_n_nrm
            sigma = m_n

        lbd = np.sqrt(S[j] * sigma)
        W[:, j] = lbd * u
        H[j, :] = lbd * v

    W[W < eps] = 0
    H[H < eps] = 0
    #ket=clusters_matrix(np.transpose(H))
    ket=clusters_matrix(W)
    #ket=ket*(W>0)
    #ket=np.sqrt(W*np.transpose(H)/Ws[:,None])
    return ket

def Mod_matrix(X,ket,Ws,thr=0.0001):
    #print('here')

    N=len(ket[:,0])
    m=len(ket[0,:])
    if (m==1):
        ket=np.ones((N,1))
    #sW=sum(Ws)
    #W=(X.dot(ket)*ket)#-np.outer(Ws,np.dot(Ws,ket)/sW)*ket
    def Dt(ket):
        return X.dot(ket)#/Ws[:,None]#X.dot(X.dot(ket))/Ws[:,None]#+X.dot(ket)#/Ws[:,None]
    W=Dt(ket)*ket#-np.outer(Ws,np.dot(Ws,ket)/sW)*ket
    #W=X.dot(ket)*ket#-np.outer(Ws,np.dot(Ws,ket)/sW)*ket
    sW=sum(W)
    #sW=np.dot(Ws,ket)

    ketb=(np.random.random((N,m))-0.5)*ket
    ketb=ketb*N/np.sqrt(sum(ketb*ketb))
    ketb_n=ketb.copy()
    kettr=ket.copy()
    c_max=0
    ketf=np.zeros((N,0))
    ketf=np.concatenate((ketf,(ketb_n>thr/N),(ketb_n<-thr/N)),axis=1)
    #ketf=ketb_n

    #print(len(ket[0,:]))
    lm=np.zeros(m)
    for i in range(c_max):
        #input(i)
        #ketb_n=X.dot(ketb_n)*kettr-np.outer(Ws,(np.dot(Ws,ketb_n)/sW))*kettr-W*ketb_n
        ketb_n=(Dt(ketb_n)*kettr-W*(sum(W*ketb_n)/sW)*kettr)
        #lm_n=sum(ketb_n*ketb)
        ketb_n=ketb_n/np.sqrt(sum(ketb_n*ketb_n))
        sm=sum((ketb_n-ketb)**2)

        lm_n=sum(X.dot(ketb_n)*ketb_n)
        #print(lm_n)
        sm=(lm_n-lm)**2

        if sum(sm<thr/N)>0:
            ind=sm<thr
            #ketb_n[:,ind][ketb_n[:,ind]<thr]=0
            ketf=np.concatenate((ketf,(ketb_n[:,ind]>thr/N),(ketb_n[:,ind]<-thr/N)),axis=1)
            ind2=(1-ind)>0
            if (sum(ind2)>0):
                #print(ind2)
                #print(len(ketb_n[1,:]))
                #print(len(W[1,:]))
                ketb_n=ketb_n[:,ind2]
                kettr=kettr[:,ind2]
                W=W[:,ind2]
                sW=sum(W)
                lm=sum(X.dot(ketb_n)*ketb_n)
                #W=W[:,ind2]
                #sW=sum(W)
                #ketb=ketb_n.copy()


            else:

                break
        else:
            lm=lm_n.copy()
            #ketb=ketb_n.copy()
        if (i==c_max):
            #ketb_n[ketb_n<thr]=0
            ketf=np.concatenate((ketf,(ketb_n>thr/N),(ketb_n<-thr/N)),axis=1)
        ketb=ketb_n.copy()

    return ketf


def maximize_modularity(X,Ws,kets):
    a=0
    Wtot=np.sum(Ws)
    md=[modularity(X,Ws,ket) for ket in kets]
    amx=np.argmax(md)
    print(amx)
    return kets[amx]

def NMF_walk_loop(X,X2,Ws,n_components,ket=[],lm=7,max_iter=10,max_iter_w=20,thr1=0.001,thr2=0.001):
    if len(ket)==0:
        model = NMF( n_components=n_components,init='nndsvd', random_state=0,max_iter=10000,solver='cd')
        ket1a = (model.fit_transform(X))
        ket1ad = clusters_matrix(ket1a)
        ket1b=(np.transpose(model.components_))
        ket1bd = clusters_matrix(ket1b)
        #ket_nmf = clusters_matrix(ket1b)
        model = NMF( n_components=n_components,init='nndsvd', random_state=0,max_iter=10000,solver='cd')#mu')
        ket2a = (model.fit_transform(X2))
        ket2ad = clusters_matrix(ket2a)
        ket2b=(np.transpose(model.components_))
        ket2bd = clusters_matrix(ket2b)
        #ket_nmf=maximize_modularity(X,Ws,[ket1a,ket1b,ket2a,ket2b,ket1ad,ket1bd,ket2ad,ket2bd])
        ket_nmf=maximize_modularity(X,Ws,[ket1ad,ket1bd,ket2ad,ket2bd])
        #ket_nmf=np.sqrt(ket1a*ket1b/Ws[:,None])
        #ket_nmf=nndsvd(X, n_components, Ws)
        #ket_nmf=clusters_matrix(model.fit_transform(X))
    #ket2=np.transpose(model.components_)

    else:
        ket_nmf=ket

    N=len(ket_nmf)
    m=n_components
    avg=X.mean()
    for iter in range(max_iter):
        print('iter='+str(iter))
        #ket_nmf_pr=ket_nmf.copy()
        if (iter==0):
            ketwl=walk_likelihood(X,Ws,ket_nmf,lm=lm,max_iter=max_iter_w)
            m=len(ketwl[0,:])
            c_identity_pr=ketwl.dot(np.array(range(m)))
        else:
            ketwl=walk_likelihood(X,Ws,ket2,lm=lm,max_iter=max_iter_w)
            m=len(ketwl[0,:])
            c_identity=ketwl.dot(np.array(range(m)))
            #if iter==1)
            if (nmi(c_identity,c_identity_pr)>0.99):
                break
            else:
                c_identity_pr=c_identity.copy()
            #if converged(ketwl,ketwl_pr,thr2):
            #    break
        ketwl_pr=ketwl.copy()
        if (iter<(max_iter-1)):
            model = NMF( n_components=n_components,init='custom', random_state=0,max_iter=1000,solver='cd')#mu')
            #ket2[ket2==0]=avg
            ket2=ketwl*Ws[:,None]
            #ket2[ket2==0]=avg
            ket2=np.transpose(ket2/np.maximum(np.sum(ketwl,axis=0),0))
            ket1=ketwl.copy()
            if (len(ket2[:,0])<n_components):
                md=n_components-len(ket2[:,0])
                ket2=np.concatenate((ket2,np.zeros((md,N))),axis=0)
                ket1=np.concatenate((ket1,np.zeros((N,md))),axis=1)
            ket1[ket1==0]=avg
            ket2[ket2==0]=avg
            ket2=np.transpose(ket1*Ws[:,None]/np.sum(ket1,axis=0))
            ket2 = model.fit_transform(X,W=ket1,H=ket2)
            #ket2=np.sqrt(ket2*np.transpose(model.components_)/Ws[:,None])
            if (np.isnan(ket2).any() or np.isinf(ket2).any()):
                break
    return ketwl

def NMF_walk_loop2(X,X2,Ws,n_comps,ket=[],lm=7,max_iter=5,max_iter_w=20,thr1=0.001,thr2=0.001):
    if len(ket)==0:
        model = NMF( n_components=n_comps,init='nndsvda', random_state=0, verbose=False,max_iter=1,solver='mu')
        ket2 = model.fit_transform(X)
        H=(model.components_)
        for cnt in range(10):
            W=(ket2*np.transpose(H)/Ws[:,None])**0.5
            H=np.transpose(W*Ws[:,None]/sum(W*W))
            model = NMF( n_components=n_comps,init='custom', random_state=0, verbose=False,max_iter=100,solver='mu')
            ket2 = model.fit_transform(X,W=W,H=H)
            H=(model.components_)
        model = NMF( n_components=n_comps,init='custom', random_state=0, verbose=False,max_iter=10000,solver='mu')
        ket2 = model.fit_transform(X,W=W,H=H)

    else:
        ket2=ket

    N=len(ket2)
    m=n_comps
    avg=X.mean()
    for iter in range(max_iter):
        #print(iter)
        #ket_nmf_pr=ket_nmf.copy()
        ketwl=walk_likelihood(X,Ws,ket2,lm=lm)
        #if not (iter==0):
        #    if converged(ketwl,ketwl_pr,thr2):
        #        break
        ketwl_pr=ketwl.copy()
        if (iter<(max_iter-1)):
            model = NMF( n_components=n_comps,init='custom', random_state=0,max_iter=1000,solver='mu')
            ket2=ketwl*Ws[:,None]
            ket2=np.transpose(ket2/np.sum(ketwl,axis=0))
            ket1=ketwl.copy()
            if (len(ket2[:,0])<n_comps):
                md=n_comps-len(ket2[:,0])
                ket2=np.concatenate((ket2,np.zeros((md,N))),axis=0)
                ket1=np.concatenate((ket1,np.zeros((N,md))),axis=1)
            ket1[ket1==0]=avg
            ket2[ket2==0]=avg
            ket2 = model.fit_transform(X,W=ket1,H=ket2)
    return ketwl

def find_clusters_walk_NMF_loop(X,X2,Ws,lm=7,max_iter=10,max_iter_w=20,thr1=0.001,thr2=0.001,m_max=20):
    ket=NMF_walk_loop(X,X2,Ws,2,lm=lm,max_iter=max_iter,max_iter_w=max_iter_w,thr1=thr1,thr2=thr2)
    mds=[modularity(X,Ws,ket)]
    kets=[ket]
    for m in range(3,m_max):
        print(m)
        #ketpr=ket.copy()
        #ket=NMF_walk_loop(X,X2,Ws,m,lm=lm,max_iter=max_iter,max_iter_w=max_iter_w,thr1=thr1,thr2=thr2)
        ket=NMF_walk_loop(X,X,Ws,m,ket=ket,lm=lm,max_iter=max_iter,max_iter_w=max_iter_w,thr1=thr1,thr2=thr2)
        md=modularity(X,Ws,ket)
        print(md)
        kets.append(ket)
        #if mds[-1]>md:
        #    ket=ketpr.copy()
        #    break
        #else:
        mds.append(md)
    amx=np.argmax(mds)
    return kets[amx]

def walk_likelihood_variable_comms_old(X,Ws,lm=7,max_iter=20,max_iter_b=50):
    #model = NMF( n_components=2,init='nndsvda', random_state=0,max_iter=1000,solver='mu')
    #ket = clusters_matrix(model.fit_transform(X))
    N=len(Ws)
    ket=nndsvd(X,2,Ws)
    ket = walk_likelihood_maximize_modularity(X,Ws,ket,lm=lm,max_iter=max_iter)
    c_identity=ket.dot(np.array(range(2)))
    for cnt in range(max_iter_b):
        print(cnt)
        m=len(ket[0,:])
        for i in range(m):
            #model = NMF( n_components=2,init='nndsvda', random_state=0,max_iter=1000,solver='mu')
            Xt=X[ket[:,i]==1,:]#[:,None]
            ket2=nndsvd(Xt,2,Ws)
            if (i==0):
                ket_new=np.zeros((N,2))
                ket_new[ket[:,i]==1,:]=ket2
            else:
                ketn=np.zeros((N,2))
                ketn[ket[:,i]==1,:]=ket2
                ket_new=np.concatenate((ket_new,ketn),axis=1)
        print(len(ket_new[0,:]))
        print(len(ket_new[:,0]))

        ket_new = walk_likelihood_maximize_modularity(X,Ws,ket_new,lm=lm,max_iter=max_iter)
        print(len(ket_new[0,:]))
        m=len(ket_new[0,:])
        c_identity_new=ket_new.dot(np.array(range(m)))
        if nmi(c_identity,c_identity_new)>0.98:
            break
        else:
            ket=ket_new.copy()
            c_identity=c_identity_new.copy()
    return ket_new


def find_active_comms(ket,ket_new, thr=0.01):
    m=len(ket[0,:])
    m_new=len(ket_new[0,:])
    Ks=2*np.dot(np.transpose(ket),ket_new)/(np.outer(sum(ket),np.ones(m_new))+np.outer(np.ones(m),sum(ket_new)))
    nac=sum(Ks>(1-thr))==0
    active_comms = np.array(range(m_new))[nac]
    inactive_comms=np.array(range(m_new))[nac==0]
    return [inactive_comms,active_comms]

def walk_likelihood_variable_comms(X,Ws,ket=[],lm=7,max_iter=20,max_iter_b=50):
    Xa=X.copy()
    Xa.setdiag(1)
    N=len(Ws)
    if len(ket)==0:
        ket=nndsvd(Xa,2,Ws)
        #model = NMF( n_components=2,init='nndsvd', random_state=0,max_iter=100000,solver='cd')
        #ket = clusters_matrix(model.fit_transform(Xa))
    ket = walk_likelihood_maximize_modularity(X,Ws,ket,lm=lm,max_iter=max_iter)
    m=len(ket[0,:])
    c_identity=ket.dot(np.array(range(m)))
    active_comms=np.array(range(m))
    ket_new=np.zeros((N,0))
    a=0
    for cnt in range(max_iter_b):
        #print(cnt)
        m=len(ket[0,:])
        for i in active_comms:#range(m):
            #model = NMF( n_components=2,init='nndsvda', random_state=0,max_iter=100000,solver='cd')
            Xt=Xa[ket[:,i]==1,:]#[:,ket[:,i]==1]#[:,None]
            #ket2 = clusters_matrix(model.fit_transform(Xt))
            ket2=nndsvd(Xt,2,Ws[ket[:,i]==1])
            #ket2 = walk_likelihood(Xt,Ws[ket[:,i]==1],ket2,lm=lm,max_iter=max_iter)
            #ket2=nndsvd(Xt,2,Ws)
            ketn=np.zeros((N,2))
            ketn[ket[:,i]==1,:]=ket2
            ket_new=np.concatenate((ket_new,ketn),axis=1)

        ket_new = walk_likelihood_maximize_modularity(X,Ws,ket_new,lm=lm,max_iter=max_iter)
        print(len(ket_new[0,:]))
        m=len(ket_new[0,:])
        c_identity_new=ket_new.dot(np.array(range(m)))
        if (len(active_comms)==0):
            break
        if max(c_identity)>max(c_identity_new):
            break

        if max(c_identity)==max(c_identity_new):
            if a==1:
                if nmi(c_identity,c_identity_new)>0.95:
                    break

            a=1
            if nmi(c_identity,c_identity_new)>0.99:
                break
        else:
            a=0

        [inactive_comms,active_comms]=find_active_comms(ket,ket_new)
        ket=ket_new.copy()
        ket_new=ket_new[:,inactive_comms]
        c_identity=c_identity_new.copy()

    print(len(ket[0,:]))
    return ket#_new

def walk_likelihood_variable_comms_with_nmf(X,Ws,ket=[],lm=7,max_iter=20,max_iter_b=50):
    Xa=X.copy()
    Xa.setdiag(1)
    N=len(Ws)
    if len(ket)==0:
        model = NMF( n_components=2,init='nndsvd', random_state=0,max_iter=100000,solver='cd')
        ket = clusters_matrix(model.fit_transform(Xa))
    ket = walk_likelihood_maximize_modularity(X,Ws,ket,lm=lm,max_iter=max_iter)
    m=len(ket[0,:])
    c_identity=ket.dot(np.array(range(m)))
    active_comms=np.array(range(m))
    ket_new=np.zeros((N,0))
    a=0
    for cnt in range(max_iter_b):
        m=len(ket[0,:])
        for i in active_comms:#range(m):
            model = NMF( n_components=2,init='nndsvd', random_state=0,max_iter=100000,solver='cd')
            Xt=Xa[ket[:,i]==1,:]#[:,ket[:,i]==1]#[:,None]
            ket2 = clusters_matrix(model.fit_transform(Xt))
            ketn=np.zeros((N,2))
            ketn[ket[:,i]==1,:]=ket2
            ket_new=np.concatenate((ket_new,ketn),axis=1)

        ket_new = walk_likelihood_maximize_modularity(X,Ws,ket_new,lm=lm,max_iter=max_iter)
        print(len(ket_new[0,:]))
        m=len(ket_new[0,:])
        c_identity_new=ket_new.dot(np.array(range(m)))
        if (len(active_comms)==0):
            break
        if max(c_identity)>max(c_identity_new):
            break

        if max(c_identity)==max(c_identity_new):
            if a==1:
                if nmi(c_identity,c_identity_new)>0.9:
                    break

            a=1
            if nmi(c_identity,c_identity_new)>0.99:
                break
        else:
            a=0

        [inactive_comms,active_comms]=find_active_comms(ket,ket_new)
        ket=ket_new.copy()
        ket_new=ket_new[:,inactive_comms]
        c_identity=c_identity_new.copy()

    print(len(ket[0,:]))
    return ket#_new

def walk_likelihood_variable_comms_null(X,Ws,lm=7,max_iter=20,max_iter_b=50):
    Xa=X.copy()
    Xa.setdiag(1)
    N=len(Ws)
    #ket=nndsvd(Xa,2,Ws)
    model = NMF( n_components=2,init='nndsvd', random_state=0,max_iter=1000,solver='cd')
    ket = clusters_matrix(model.fit_transform(Xa))
    #ket = walk_likelihood_maximize_modularity(X,Ws,ket,lm=lm,max_iter=max_iter)
    c_identity=ket.dot(np.array(range(2)))
    active_comms=[0,1]
    ket_new=np.zeros((N,0))
    a=0
    for cnt in range(max_iter_b):
        #print(cnt)
        m=len(ket[0,:])
        for i in active_comms:#range(m):
            model = NMF( n_components=2,init='nndsvd', random_state=0,max_iter=1000,solver='cd')
            Xt=Xa[ket[:,i]==1,:]#[:,None]
            ket2 = clusters_matrix(model.fit_transform(Xt))
            #ket2=nndsvd(Xt,2,Ws)
            ketn=np.zeros((N,2))
            ketn[ket[:,i]==1,:]=ket2
            ket_new=np.concatenate((ket_new,ketn),axis=1)

        ket_new = walk_likelihood_maximize_modularity(X,Ws,ket_new,lm=0,max_iter=0)#max_iter)
        #print(len(ket_new[0,:]))
        m=len(ket_new[0,:])
        c_identity_new=ket_new.dot(np.array(range(m)))
        if (len(active_comms)==0):
            break
        if max(c_identity)>max(c_identity_new):
            break
        if max(c_identity)==max(c_identity_new):
            if a==1:
                if nmi(c_identity,c_identity_new)>0.9:
                    break

            a=1
            if nmi(c_identity,c_identity_new)>0.98:
                break
        else:
            a=0

        [inactive_comms,active_comms]=find_active_comms(ket,ket_new)
        ket=ket_new.copy()
        ket_new=ket_new[:,inactive_comms]
        c_identity=c_identity_new.copy()

    print(len(ket[0,:]))
    return ket#_new

def walk_likelihood_variable_comms_mod(X,Ws,ket=[],lm=7,max_iter=20,max_iter_b=50):
    Xa=X.copy()
    Xa.setdiag(1)
    N=len(Ws)
    if len(ket)==0:
        ket=Mod_matrix(X,np.ones((N,1)),Ws)
    ket = walk_likelihood_maximize_modularity(X,Ws,ket,lm=lm,max_iter=max_iter,twice=1)
    m=len(ket[0,:])
    c_identity=ket.dot(np.array(range(m)))
    active_comms=np.array(range(m))
    ket_new=np.zeros((N,0))
    a=0
    inactive_comms=[]
    for cnt in range(max_iter_b):
        print(cnt)
        m=len(ket[0,:])
        print(m)
        ketf=Mod_matrix(X,ket[:,active_comms],Ws)
        la=len(active_comms)
        ket_new=np.concatenate((ket[:,inactive_comms],ketf),axis=1)
        ket_new = walk_likelihood_maximize_modularity(X,Ws,ket_new,lm=lm,max_iter=max_iter,twice=1)
        print(len(ket_new[0,:]))
        m=len(ket_new[0,:])
        c_identity_new=ket_new.dot(np.array(range(m)))
        if (len(active_comms)==0):
            break
        if max(c_identity)>max(c_identity_new):
            break

        if max(c_identity)==max(c_identity_new):
            if a==1:
                if nmi(c_identity,c_identity_new)>0.95:
                    break

            a=1
            if nmi(c_identity,c_identity_new)>0.99:
                break
        else:
            a=0

        [inactive_comms,active_comms]=find_active_comms(ket,ket_new)
        #m2=len(ket_new[0,:])
        #[inactive_comms,active_comms]=[[],list(range(m2))]
        if (len(active_comms)==0):
            break
        ket=ket_new.copy()
        ket_new=ket_new[:,inactive_comms]
        c_identity=c_identity_new.copy()

    print(len(ket[0,:]))
    return ket#_new

def walk_likelihood_variable_comms_rand(X,Ws,ket=[],lm=7,max_iter=20,max_iter_b=50,twice=0):
    N=len(Ws)
    if len(ket)==0:
        ket=Mod_matrix(X,np.ones((N,1)),Ws)
    ket = walk_likelihood_maximize_modularity(X,Ws,ket,lm=lm,max_iter=max_iter,twice=twice)
    m=len(ket[0,:])
    c_identity=ket.dot(np.array(range(m)))
    active_comms=np.array(range(m))
    ket_new=np.zeros((N,0))
    a=0
    inactive_comms=[]
    for cnt in range(max_iter_b):
        print(cnt)
        m=len(ket[0,:])
        print(m)
        ket_new=ket.copy()
        ket2=ket[:,active_comms]*np.random.randint(2,size=N)[:,None]
        ket_new[:,active_comms]=ket_new[:,active_comms]-(ket2)
        ket_new=np.concatenate((ket_new,ket2),axis=1)
        #ketf=Mod_matrix(X,ket[:,active_comms],Ws)
        #la=len(active_comms)
        #ket_new=np.concatenate((ket[:,inactive_comms],ketf),axis=1)
        ket_new = walk_likelihood_maximize_modularity(X,Ws,ket_new,lm=lm,max_iter=max_iter,twice=twice)
        print(len(ket_new[0,:]))
        m=len(ket_new[0,:])
        c_identity_new=ket_new.dot(np.array(range(m)))
        if (len(active_comms)==0):
            break
        if max(c_identity)>max(c_identity_new):
            break

        if max(c_identity)==max(c_identity_new):
            if a==1:
                if nmi(c_identity,c_identity_new)>0.95:
                    break

            a=1
            if nmi(c_identity,c_identity_new)>0.99:
                break
        else:
            a=0

        [inactive_comms,active_comms]=find_active_comms(ket,ket_new)
        if (len(active_comms)==0):
            break
        ket=ket_new.copy()
        ket_new=ket_new[:,inactive_comms]
        c_identity=c_identity_new.copy()

    print(len(ket[0,:]))
    return ket#_new

from sklearn.cluster import SpectralClustering
def walk_likelihood_variable_comms_with_sp(X,Ws,ket=[],lm=7,max_iter=20,max_iter_b=50):
    Xa=X.copy()
    Xa.setdiag(1)
    N=len(Ws)
    if len(ket)==0:
        model = SpectralClustering( n_clusters=2,assign_labels='discretize',affinity='precomputed').fit(X)#,assign_labels='discretize'
        lb=model.labels_
        ket = np.concatenate((lb[:,None],(lb==0)[:,None]),axis=1)
    ket = walk_likelihood_maximize_modularity(X,Ws,ket,lm=lm,max_iter=max_iter)
    m=len(ket[0,:])
    c_identity=ket.dot(np.array(range(m)))
    active_comms=np.array(range(m))
    ket_new=np.zeros((N,0))
    a=0
    for cnt in range(max_iter_b):
        m=len(ket[0,:])
        for i in active_comms:#range(m):
            model = SpectralClustering( n_clusters=2,assign_labels='discretize',affinity='precomputed').fit(Xa[ket[:,i]==1,:][:,ket[:,i]==1])
            lb=model.labels_
            ket2 = np.concatenate((lb[:,None],(lb==0)[:,None]),axis=1)
            ketn=np.zeros((N,2))
            ketn[ket[:,i]==1,:]=ket2
            ket_new=np.concatenate((ket_new,ketn),axis=1)

        ket_new = walk_likelihood_maximize_modularity(X,Ws,ket_new,lm=lm,max_iter=max_iter)
        print(len(ket_new[0,:]))
        m=len(ket_new[0,:])
        c_identity_new=ket_new.dot(np.array(range(m)))
        if (len(active_comms)==0):
            break
        if max(c_identity)>max(c_identity_new):
            break

        if max(c_identity)==max(c_identity_new):
            if a==1:
                if nmi(c_identity,c_identity_new)>0.9:
                    break

            a=1
            if nmi(c_identity,c_identity_new)>0.99:
                break
        else:
            a=0

        [inactive_comms,active_comms]=find_active_comms(ket,ket_new)
        ket=ket_new.copy()
        ket_new=ket_new[:,inactive_comms]
        c_identity=c_identity_new.copy()

    print(len(ket[0,:]))
    return ket#_new

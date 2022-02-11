import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi
from sklearn.decomposition import NMF

def find_active_comms(U,U_new, thr=0.01):
    m=len(U[0,:])
    m_new=len(U_new[0,:])
    Ks=2*np.dot(np.transpose(U),U_new)/(np.outer(sum(U),np.ones(m_new))+np.outer(np.ones(m),sum(U_new)))
    nac=sum(Ks>(1-thr))==0
    active_comms = np.array(range(m_new))[nac]
    inactive_comms=np.array(range(m_new))[nac==0]
    return [inactive_comms,active_comms]

class walk_likelihood:
    def __init__(self, X):
        self.X=X
        self.N=X.shape[0]
        self.w=np.zeros(self.N)
        print(self.w.shape)
        self.w[:]=X.dot(np.ones(self.N))

    def initialize_WLA(self,init,m):
        if init=='random':
            self.clusters=np.random.randint(m,size=self.N)
        elif init=='NMF':
            model = NMF( n_components=m,init='nndsvda',solver='mu')
            self.clusters=np.argmax(model.fit_transform(self.X),axis=1)
        self.U=np.zeros((self.N,m))
        self.m=m
        self.U[range(self.N),self.clusters]=1


    def WLA(self,U=[],clusters=[],init='NMF',m=0,lm=8,max_iter_WLA=20,thr_WLA=0.99,eps=0.00000001):
        retuns_options='U'
        if not len(U) ==0:
            self.U=U
            self.clusters=np.argmax(U,axis=1)
            self.m=U.shape[1]
        elif not len(clusters) ==0:
            retuns_options='clusters'
            self.clusters=clusters-np.min(a)
            self.m=np.max(self.clusters)+1
            self.U=np.zeros((self.N,self.m))
            self.U[range(self.N),self.clusters]=1
        else:
            self.initialize_WLA(init,m)
        clusters_prev=self.clusters
        for iter in range(max_iter_WLA):
            nz_values=sum(self.U)>0
            if (np.prod(nz_values)==0):
                self.U=self.U[:,nz_values]
                self.m=U.shape[1]
            dV=self.X.dot(self.U).astype(float)
            V=dV
            for i in range(lm-1):
                dV=self.X.dot(dV/self.w[:,None])
                V=V+dV
            Q=V.T.dot(self.U)/self.w.dot(self.U)
            g=1/np.diagonal(Q)
            log_Q=np.log(Q+eps*(Q==0))
            F=np.dot(V,log_Q*g[:,None])-np.outer(self.w,sum(Q*g[:,None]))
            self.clusters=np.argmax(F,axis=1)
            self.U=np.zeros((self.N,self.m))
            self.U[range(self.N),self.clusters]=1
            if nmi(self.clusters,clusters_prev)>thr_WLA:
                break
            clusters_prev=self.clusters

        nz_values=sum(self.U)>0
        if (np.prod(nz_values)==0):
            self.U=self.U[:,nz_values]
            self.m=U.shape[1]
        if retuns_options=='U':
            return self.U
        else:
            return self.clusters

    def merge_communities(self):
        W=np.dot(np.transpose(self.w),self.U)
        M=np.dot(np.transpose(self.U),self.X.dot(self.U))-np.outer(W,W)/sum(W)
        np.fill_diagonal(M,0)
        merge=0
        if (np.sum(M>0)>0):
            merge=1
            amx=np.argmax(M)
            i=int(amx/self.m)
            j=amx-i*self.m
            self.U[:,i]+=self.U[:,j]
            self.m=self.U.shape[1]
            self.U=self.U[:,np.array(list(range(j))+list(range(j+1,self.m)))]
        self.m=self.U.shape[1]
        return merge

    def bifuraction(self,active_comms,bifuraction_type):

        if bifuraction_type=='random':
            U2=self.U[:,active_comms]*np.random.randint(2,size=self.N)[:,None]

        elif bifuraction_type=='NMF':
            U2=np.zeros((self.N,len(active_comms)))
            i=0
            for c in active_comms:
                X2=X[self.U[:,c]==1,:]
                model = NMF( n_components=2,init='nndsvda',solver='mu')
                U2[self.U[:,c]==1,i]=np.argmax(model.fit_transform(X2),axis=1)
                i+=1

        self.U[:,active_comms]=self.U[:,active_comms]-(U2)
        self.U=np.concatenate((self.U,U2),axis=1)

    def WLCF(self,max_iter_WLCF=50,U=[],thr_WLCF=0.99,bifuraction_type='random',**params):
        if len(U)==0:
            self.U=np.ones((self.N,1))
            self.m=1
        m_prev=self.m
        clusters_prev=np.argmax(self.U,axis=1)
        active_comms=np.array(range(self.m))
        inactive_comms=[]
        for iter in range(max_iter_WLCF):
            U_prev=self.U.copy()
            self.bifuraction(active_comms,bifuraction_type)
            merge=1
            while(merge):
                self.WLA(U=self.U,**params)
                merge=self.merge_communities()

            if (nmi(clusters_prev,self.clusters)>thr_WLCF) and m_prev==self.m:
                break

            [inactive_comms,active_comms]=find_active_comms(U_prev,self.U)
            if (len(active_comms)==0):
                break
            U_prev=U.copy()
            clusters_prev=self.clusters.copy()
            m_prev=self.m
        return self.U

    def find_communities(X):
        self.X=X
        if self.method=='WLA':
            self.WLA()
        elif self.method=='WLCF':
            self.WLCF()

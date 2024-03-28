import numpy as np

class HornPoseFitting:
    def __init__(self):
        super(HornPoseFitting,self).__init__()
        
    def rotate(self,a,i,j,k,l,s,tau):
        g=a[i][j]
        h=a[k][l]
        a[i][j]=g-s*(h+g*tau)
        a[k][l]=h+s*(g-h*tau)

    def myjacobi(self,a,n,d,v,nrot):
        b=np.zeros((5))
        z=np.zeros((5))

        for ip in range(1,n+1):
            for iq in range(1,n+1):
                v[ip][iq]=0.0
            v[ip][ip]=1.0

        for ip in range(1,n+1):
            b[ip]=d[ip]=a[ip][ip]
            z[ip]=0.0
        nrot=0
        for i in range(1,51):
            sm=0.0
            for ip in range(1,n):
                for iq in range(1,n+1):
                    sm += abs(a[ip][iq])
            if (sm == 0.0):
                return
            if (i < 4):
                tresh=0.2*sm/(n*n)
            else:
                tresh=0.0
            for ip in range(1,n):
                for iq in range(ip+1,n+1):
                    g=100.0*abs(a[ip][iq])
                    if (i > 4 and abs(d[ip])+g == abs(d[ip]) and abs(d[iq])+g == abs(d[iq])):
                        a[ip][iq]=0.0
                    elif (abs(a[ip][iq]) > tresh):
                        h=d[iq]-d[ip]
                        if (abs(h)+g == abs(h)):
                            t=(a[ip][iq])/h
                        else:
                            theta=0.5*h/(a[ip][iq])
                            t=1.0/(abs(theta)+np.sqrt(1.0+theta*theta))
                            if (theta < 0.0):
                                t = -t
                        c=1.0/np.sqrt(1+t*t)
                        s=t*c
                        tau=s/(1.0+c)
                        h=t*a[ip][iq]
                        z[ip] -= h
                        z[iq] += h
                        d[ip] -= h
                        d[iq] += h
                        a[ip][iq]=0.0
                        for j in range(1,ip):
                            self.rotate(a,j,ip,j,iq,s,tau)
                        for j in range(ip+1,iq):
                            self.rotate(a,ip,j,j,iq,s,tau)
                        for j in range(iq+1,n+1):
                            self.rotate(a,ip,j,iq,j,s,tau)
                        for j in range(1,n+1):
                            self.rotate(v,j,ip,j,iq,s,tau)
                        nrot+=1
            for ip in range(1,n+1):
                b[ip] += z[ip]
                d[ip]=b[ip]
                z[ip]=0.0


    def lmshorn(self,P1, P2,n, A):
        C1=np.zeros((3))
        C2=np.zeros((3))
        Sxx=0.0
        Sxy=0.0
        Sxz=0.0
        Syx=0.0
        Syy=0.0
        Syz=0.0
        Szx=0.0
        Szy=0.0
        Szz=0.0
        N=np.zeros((5,5))
        D=np.zeros((5))
        V=np.zeros((5,5))
        R=np.zeros((3,3))
        T=np.zeros((3))
        nrot=0
        #calculate centroids
        for i in range(0,n):
            for j in range(0,3):
                C1[j] += P1[i][j]
                C2[j] += P2[i][j]

        for j in range(0,3):
            C1[j] /= n
            C2[j] /= n

        #translate point sets to cetroids
        for i in range(0,n):
            for j in range(0,3):
                P1[i][j] -= C1[j]
                P2[i][j] -= C2[j]

        #calculate elements of M matrix
        for i in range(0,n):
            Sxx += P1[i][0]*P2[i][0]
            Sxy += P1[i][0]*P2[i][1]
            Sxz += P1[i][0]*P2[i][2]

            Syx += P1[i][1]*P2[i][0]
            Syy += P1[i][1]*P2[i][1]
            Syz += P1[i][1]*P2[i][2]

            Szx += P1[i][2]*P2[i][0]
            Szy += P1[i][2]*P2[i][1]
            Szz += P1[i][2]*P2[i][2]

        #generate N matrix

        #Note : matrix indeces start at 1, to be NR compatible

        N[1][1], N[1][2], N[1][3], N[1][4] = Sxx+Syy+Szz, Syz-Szy, Szx-Sxz, Sxy-Syx
        N[2][1], N[2][2], N[2][3], N[2][4] = Syz-Szy, Sxx-Syy-Szz, Sxy+Syx, Szx+Sxz
        N[3][1], N[3][2], N[3][3], N[3][4] = Szx-Sxz, Sxy+Syx, -Sxx+Syy-Szz, Syz+Szy
        N[4][1], N[4][2], N[4][3], N[4][4] = Sxy-Syx, Szx+Sxz, Syz+Szy, -Sxx-Syy+Szz


        #find eigenvectors
        self.myjacobi(N,4,D,V,nrot)

        #find max eigenvalue
        max_eigind = 1
        max_eigval = D[max_eigind]
        for i in range(2,5):
            if (D[i] > max_eigval):
                max_eigind = i
                max_eigval = D[max_eigind]

        #generate optimal rotation matrix
        #Note : start matrix indexing at 0 again
        q0 = V[1][max_eigind]
        q1 = V[2][max_eigind]
        q2 = V[3][max_eigind]
        q3 = V[4][max_eigind]

        R[0][0] = q0*q0 + q1*q1 - q2*q2 - q3*q3
        R[0][1] = 2*(q1*q2 - q0*q3)
        R[0][2] = 2*(q1*q3 + q0*q2)
        R[1][0] = 2*(q1*q2 + q0*q3)
        R[1][1] = q0*q0 + q2*q2 - q1*q1 -q3*q3
        R[1][2] = 2*(q2*q3 - q0*q1)
        R[2][0] = 2*(q1*q3 - q0*q2)
        R[2][1] = 2*(q2*q3 + q0*q1)
        R[2][2] = q0*q0 + q3*q3 - q1*q1 -q2*q2

        #calculate translation
        T[0] = C2[0] - (R[0][0]*C1[0] + R[0][1]*C1[1] + R[0][2]*C1[2])
        T[1] = C2[1] - (R[1][0]*C1[0] + R[1][1]*C1[1] + R[1][2]*C1[2])
        T[2] = C2[2] - (R[2][0]*C1[0] + R[2][1]*C1[1] + R[2][2]*C1[2])

        #compose into matrix
        for i in range(0,3):
            for j in range(0,3):
                A[i][j] = R[i][j]

        for j in range(0,3):
            A[j][3] = T[j]
            A[3][j] = 0.0

        A[3][3] = 1.0

        #restore original point sets
        for i in range(0,n):
            for j in range(0,3):
                P1[i][j] += C1[j]
                P2[i][j] += C2[j]


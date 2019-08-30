#密文BP非线性；需要调BP里的几个参数；VHE有漏洞，且参数不合适如w
import numpy as np 
import time

x =np.array([[-3,-2.7,-2.4,-2.1,-1.8,-1.5,-1.2,-0.9,-0.6,-0.3,0,0.3,0.6,0.9,1.2,1.5,1.8],
            [-2,-1.8,-1.6,-1.4,-1.2,-1,-0.8,-0.6,-0.4,-0.2,-2.2204,0.2,0.4,0.6,0.8,1,1.2]])
x=x.T
y  =np.array([[0.6589,0.2206,-0.1635,-0.4712,-0.6858,-0.7975,-0.8040, -0.7113,-0.5326,-0.2875,0,0.3035,0.5966,0.8553,1.0600,1.1975,1.2618]])
y=y.T

x=x*10000
y=y*10000#放大一万倍变成整数以用VHE加密

def get_T(m):
    T = (10 * np.random.rand(m, 1)).astype('int')
    return T

def get_c_star(c,m,l):
    c_star_list=list()
    for j in range(line_number):
        c_star=np.zeros(l*m,dtype='int') 
        for i in range(m):
            temp=int(c[j][i])
            b=np.array(list(np.binary_repr(np.abs(temp))),dtype='int')
            if(c[j][i]<0):
                b*=-1
            c_star[(i*l)+(l-len(b)):(i+1)*l]+=b
        c_star_list.append(c_star)
    c_star_list=np.array(c_star_list)
    return c_star_list

def get_S_star(S,m,l):
    S_star=list() 
    for i in range(l): 
        S_star.append(S*2**(l-i-1))
    S_star=np.array(S_star).transpose(1,2,0).reshape((m,m*l))
    return S_star

def switch_key(c, S, m, T):#现在括号里的参数c是x*w;S是m*m单位阵
    l = int(np.ceil(np.log2(np.max(np.abs(c)))))#l是比特化参数，与训练数据的最大的绝对值有关，计算后输出，预测程序也要用；
    print("VHE加密时参数l值为"+str(l))
    c_star_list = get_c_star(c, m, l) 
    S_star = get_S_star(S, m, l) 
    A = (np.random.rand(1, m * l) * 10).astype('int') 
    E = (1.1 * np.random.rand(S_star.shape[0], S_star.shape[1])).astype('int')
    M = np.concatenate(((S_star - T.dot(A) + E), A), 0)

    c_prime_list=list()
    for i in range(line_number):#分别对每一行数据进行加密
        c_prime=M.dot(c_star_list[i]) 
        c_prime_list.append(c_prime)
    c_prime_list=np.array(c_prime_list)
    c_prime=c_prime_list.reshape(17,3)
    return M,c_prime,S_star

def encrypt_via_switch(x, w, m,  T):
    M,c,S_star = switch_key(x * w, np.eye(m), m, T)
    return M,c,S_star

m=x.shape[1] 
line_number=x.shape[0]

w=16
T=get_T(m) 
M,c,S_star=encrypt_via_switch(x,w,m,T)#计算公钥M和密文c

#print("c=",c)

def wucha(wih,bh,who,bo):
    hi=c.dot(wih)+bh.T
    ho=f(hi/400)*400#这里的400是参数，根据数据集等调，这里400可能不是最好的
    yi=ho.dot(who)+bo.T
    yo=(yi)
    err=(yo-y)/10000
    result=err.T.dot(err)/34
    return result,yo

def f(x,deriv=False):
    y=np.tanh(x)
    if (deriv == True):
        return (1-y**2)
    return (y)
    
def train(x,y):
    x_num=x.shape[0]#x_num条数据训练
    x_len=x.shape[1]#每条训练数据有x_len维属性
    out_num=1#最后的输出是1维
    hid_num=3#隐层有一层，3个结点
    wih=0.001*np.random.random((x_len,hid_num))#输入层和隐层间权值矩阵;
    print("wih=",wih)
    who=0.01*np.random.random((hid_num,out_num))#隐层和输出层间权值矩阵
    bh=np.zeros(hid_num)#隐层3个结点的阈值
    bh=bh.reshape(3,1)
    bo=np.zeros(out_num)#输出层1个结点的阈值
    bo=bo.reshape(1,1)
    
    #print("训练时，c[10]="+str(x[10]))
        
    a=0.0000001#学习率
    for i in range(5000):       
        hi=x.dot(wih)+bh.T
        print("第"+str(i)+"次")
        ho=f(hi/400)*400#这里的400是参数，根据数据集等调，这里400可能不是最好的
        yi=ho.dot(who)+bo.T
        yo=(yi)
               
        f_yo=np.ones((17,1))
        f_ho=f(hi/400,True)#这里的400是参数，要调
        
 
        err=y-yo#这里是真实值减预测值
        
        delta=err*f_yo
        who+=a*ho.T.dot(delta)
        
        hid_delta=f_ho*np.dot(delta,who.T)/400000000#这里的4亿是参数，要调
        wih+=a*x.T.dot(hid_delta)
        bo=bo+a*delta.T.dot(f_yo)
        bh=bh+a*hid_delta.T.dot(f_yo)
        
        wucha_result,yo=wucha(wih,bh,who,bo)
        print("训练误差："+str(wucha_result[0][0]))
        error.append(wucha_result[0][0])
                
    print("预测值为："+str(yo/10000))
    np.savetxt("yo_vhe.txt",yo/10000)
    
    np.savetxt("wih.txt",wih)
    np.savetxt("who.txt",who)
    np.savetxt("bo.txt",bo)
    np.savetxt("bh.txt",bh)
    return wih,bh,who,bo

error=[]
wih,bh,who,bo=train(c,y)
error=np.array(error)
np.savetxt("error_vhe.txt",error)

#VHE加密跑；这里的VHE有漏洞，且参数不合适，如w值；这里的BP同明文下BP
import numpy as np 
import time

mean=22.62061855670103#输出数据那一列的均值
std=7.79504576268#输出数据那一列的标准差

num_traindata=351#训练数据条数
num_predictdata=37#预测数据条数

def get_T(m):#返回T
    T = (10 * np.random.rand(m, 1)).astype('int')
    return T

def get_c_star(c,m,l):#比特化
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

def get_S_star(S,m,l):#返回I*
    S_star=list() 
    for i in range(l): 
        S_star.append(S*2**(l-i-1))
    S_star=np.array(S_star).transpose(1,2,0).reshape((m,m*l))
    return S_star

def switch_key(c, S, m, T):#现在括号里的参数c是x*w;S是m*m单位阵
    l = int(np.ceil(np.log2(np.max(np.abs(c)))))#l是比特化参数，与训练数据的最大的绝对值有关，计算后输出，预测程序也要用
    print("VHE加密时参数l值为"+str(l))
    c_star_list = get_c_star(c, m, l) #比特化
    S_star = get_S_star(S, m, l) #单位阵变成I* 
    A = (np.random.rand(1, m * l) * 10).astype('int') 
    E = (1.1 * np.random.rand(S_star.shape[0], S_star.shape[1])).astype('int')
    M = np.concatenate(((S_star - T.dot(A) + E), A), 0)

    c_prime_list=list()
    for i in range(line_number):#分别对每一行数据进行加密
        c_prime=M.dot(c_star_list[i]) 
        c_prime_list.append(c_prime)
    c_prime_list=np.array(c_prime_list)
    c_prime=c_prime_list.reshape(num_traindata,14)#原数据的输入有13维，加密后变为14维
    return M,c_prime,S_star 

def encrypt_via_switch(x, w, m,  T):
    M,c,S_star = switch_key(x * w, np.eye(m), m, T)
    return M,c,S_star

x=np.loadtxt("x2.txt")

m=x.shape[1]
line_number=x.shape[0]

w=16
T=get_T(m) #T
M,c,S_star=encrypt_via_switch(x,w,m,T)#计算公钥M和密文c
np.savetxt("M.txt",M) 

y=np.loadtxt("y2.txt")#y是训练时的期望输出
y=y.reshape(num_traindata,1)
real=y/100000*std+mean#训练数据的（真实）输出

##############################################################预测部分
x2=np.loadtxt("x3.txt")#归一化后再z标准化的预测数据

#用M.txt加密X得到c
line_number=x2.shape[0]
l=24#比特化参数，要和上面产生的l一致
w=16#VHE的参数，不合适
c_star_list = get_c_star(x2*w, m, l)
c_star_list=np.array(c_star_list)

#M=np.loadtxt("M.txt")
c_prime_list=list()
for i in range(line_number):
    c_prime=M.dot(c_star_list[i])
    c_prime_list.append(c_prime)
c_prime_list=np.array(c_prime_list)
c_prime=c_prime_list.reshape(num_predictdata,14)
c2=np.array(c_prime)   #预测数据加密后结果 

y2=np.loadtxt("y3.txt")
y2=y2.reshape(num_predictdata,1)#y2是预测数据的期望输出
real2=y2/100000*std+mean #real2预测数据的真实结果


def wucha(wih,bh,who,bo):#训练误差
    hi=c.dot(wih)+bh.T
    ho=hi
    yi=ho.dot(who)+bo.T
    yo=yi/100000*std+mean
    err=yo-real
    result=err.T.dot(err)/(num_traindata*2)
    return result

def yuce_wucha(wih,bh,who,bo):#预测误差
    hi=c2.dot(wih)+bh.T
    ho=hi
    yi=ho.dot(who)+bo.T
    yo=yi/100000*std+mean
    err=yo-real2
    result=err.T.dot(err)/(num_predictdata*2)
    return result,yo

def train(x,y):
    x_num=x.shape[0]#x_num条数据训练
    x_len=x.shape[1]#每条训练数据有x_len维属性
    out_num=1#最后的输出是1维
    hid_num=6#隐层有一层，6个结点
    wih=0.0001*np.random.random((x_len,hid_num))#输入层和隐层间权值矩阵;
    who=0.0001*np.random.random((hid_num,out_num))#隐层和输出层间权值矩阵
    bh=np.zeros(hid_num)#隐层6个结点的阈值
    bh=bh.reshape(hid_num,1)
    bo=np.zeros(out_num)#输出层1个结点的阈值
    bo=bo.reshape(out_num,1)
    
    a=0.000000000000001#学习率；
    isFirst=0 
    for i in range(250):   
        if(isFirst==0):
            start =time.clock()
            isFirst=1
        
        hi=x.dot(wih)+bh.T
        print("第"+str(i)+"次")
        ho=hi
        yi=ho.dot(who)+bo.T
        yo=yi
        
        f_yo=np.ones((num_traindata,1))#激励函数是y=x，所以导函数是全1
        f_ho=np.ones((num_traindata,6))
        
        err=y-yo#这里是真实值减预测值
        delta=err*f_yo
        who+=a*ho.T.dot(delta)
        
        hid_delta=f_ho*np.dot(delta,who.T)
        wih+=a*x.T.dot(hid_delta)
        
        bo=bo+a*delta.T.dot(f_yo)
        bh=bh+a*hid_delta.T.dot(f_yo)
        
        wucha_result=wucha(wih,bh,who,bo)
        print("训练误差："+str(wucha_result[0][0]))
        yuce_wucha_result,yuce2=yuce_wucha(wih,bh,who,bo)
        print("预测误差："+str(yuce_wucha_result[0][0]))
        
        if ((i+1)%25==0):
            end = time.clock()
            print('Running time: %s Seconds'%(end-start))
            timerecord.append(end-start)
        
        trainerrhistory.append(wucha_result[0][0])
        yuceerrhistory.append(yuce_wucha_result[0][0])
       
    #np.savetxt("who.txt",who)
    #np.savetxt("wih.txt",wih)
    #np.savetxt("bh.txt",bh)
    #np.savetxt("bo.txt",bo)
    print("预测结果："+str(yuce2))
    np.savetxt("prediction_vhe.txt",yuce2)
    return wih,bh,who,bo

trainerrhistory=[]
yuceerrhistory=[]

timerecord=[]
wih,bh,who,bo=train(c,y)

trainerrhistory=np.array(trainerrhistory)
yuceerrhistory=np.array(yuceerrhistory)
np.savetxt("trainerrhistory_vhe0.txt",trainerrhistory)
np.savetxt("yuceerrhistory_vhe0.txt",yuceerrhistory)
timerecord=np.array(timerecord)
for i in range(10):
    timerecord[i]='{:.5f}'.format(timerecord[i])
print(timerecord)
np.savetxt("timerecord_vhe0.txt",timerecord)

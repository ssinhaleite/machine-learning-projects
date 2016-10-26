import numpy as np
import nibabel
import logging as lg

#lg.basicConfig(filename='./prog_8_block.log',level=lg.INFO)

n_train=np.int64(278)
n_test=np.int64(138)

m=8*8*8

def get_filename(i):
    global n_train
    if i<n_train:
        return './data/set_train/train_' + str(i + 1) + '.nii'
    else:
        return './data/set_test/test_' + str(i + 1 - n_train) + '.nii'

def evaluate(A,b,X):
    return (X.dot(A)+b)

def ridge_regression(X,y,A_lam,b_lam):
#    lg.info('ridge regression: preparing data')
    print('ridge regression: preparing data')
    n,m=X.shape
    Xext = np.zeros( (n+m+1,m+1) ,dtype=np.float64 )
    Xext[0:n,0:m]=X
    Xext[0:n, m]=1
    for j in range(m):
        Xext[n+j,j]=np.sqrt(n*A_lam)
    Xext[n+m,m]=np.sqrt(n*b_lam)
    yext=np.zeros( (n+m+1) ,dtype=np.float64 )
    yext[0:n]=y

#    lg.info('ridge regression: least square')
    print('ridge regression: least square')
    gamma, residues, rank, s = np.linalg.lstsq( Xext , yext)

    A=gamma[0:m]
    b=gamma[m]

    return A,b

#lg.info('dimension reduction using means of 8*8*8 blocks')
print('dimension reduction using means of 8*8*8 blocks')

X=np.zeros( (n_train+n_test,22*26*22) )

for i in range(n_train+n_test):
#    lg.info('processing file '+str(i+1))
    print('processing file '+str(i+1))
    img_data_int16 = nibabel.load(get_filename(i)).get_data()[:, :, :, 0]
    img_data_float64 = img_data_int16.astype(np.float64)
    for x in range(22):
        for y in range(26):
            for z in range(22):
                img_chunk = img_data_float64[(8 * x):(8 * x + 8), (8 * y):(8 * y + 8), (8 * z):(8 * z + 8)]
                img_chunk_ravel = img_chunk.ravel()
                X[i, 26 * 22 * x + 22 * y + z]=np.var(img_chunk_ravel)

#lg.info('reading labels file')
print('reading labels file')
targets=[]
targets_file=open('./data/targets.csv','r')
for line in targets_file:
    targets.append(int(line))
targets_file.close()
labels = np.array(targets[0:n_train])

#lg.info('regression')
print('regression')

A_lam = 2000.0
b_lam = 0.1

A, b = ridge_regression(X[0:n_train, :], labels[0:n_train], A_lam, b_lam)

#lg.info('evaluation')
print('evaluation')

y = evaluate(A, b, X[n_train:(n_train+n_test), :])

#lg.info('saving submission')
print('saving submission')

y_=np.rint(y).astype(int)
f=open('./submission_8_block_2410.csv','w')
f.write('ID,Prediction\n')
for i in range(n_test):
    f.write(str(i+1)+','+str(y_[i])+'\n')
f.close()





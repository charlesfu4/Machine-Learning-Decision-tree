import monkdata as data
import dtree as fun
import drawtree_qt5 as draw
import random
import matplotlib.pyplot as plt
import numpy as np

def partition(data, fraction):
    ldata = list(data)
    random.shuffle(ldata)
    breakPoint = int(len(ldata)*fraction)
    return ldata[:breakPoint], ldata[breakPoint:]

"entropy calculation of training datasets"
print('Entropy of Monk-1')
print(fun.entropy(data.monk1))
print('Entropy of Monk-2')
print(fun.entropy(data.monk2))
print('Entropy of Monk-3')
print(fun.entropy(data.monk3))

"infomation gain calculation of each attribute"
print('IG of Monk-1')
for i in range(0,6):
    print(fun.averageGain(data.monk1,data.attributes[i]))
print('IG of Monk-2')
for j in range(0,6):
    print(fun.averageGain(data.monk2,data.attributes[j]))
print('IG of Monk-3')
for k in range(0,6):
    print(fun.averageGain(data.monk3,data.attributes[k]))

"split monk1 data into subsets"
print('split under the attribute-a5')

print('value = 1')
a = fun.select(data.monk1,data.attributes[4],1)
for i in range(0,6):
    print(fun.averageGain(a,data.attributes[i]))
print('value = 2')
b = fun.select(data.monk1,data.attributes[4],2)
for i in range(0,6):
    print(fun.averageGain(b,data.attributes[i]))
print('value = 3')
c = fun.select(data.monk1,data.attributes[4],3)
for i in range(0,6):
    print(fun.averageGain(c,data.attributes[i]))
print('value = 4')
d = fun.select(data.monk1,data.attributes[4],4)
for i in range(0,6):
    print(fun.averageGain(d,data.attributes[i]))


"build tree process"
print('E_train')
tree1=fun.buildTree(data.monk1,data.attributes)
print(1-fun.check(tree1,data.monk1))

tree2=fun.buildTree(data.monk2,data.attributes)
print(1-fun.check(tree2,data.monk2))

tree3=fun.buildTree(data.monk3,data.attributes)
print(1-fun.check(tree3,data.monk3))


print('E_test')
Ttree1=fun.buildTree(data.monk1,data.attributes)
print(1-fun.check(Ttree1,data.monk1test))

Ttree2=fun.buildTree(data.monk2,data.attributes)
print(1-fun.check(Ttree2,data.monk2test))

Ttree3=fun.buildTree(data.monk3,data.attributes)
print(1-fun.check(Ttree3,data.monk3test))


"tree pruning process"
def pruning(dataset,testing):
    fracs=[0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    p_error=[]
    for frac in fracs:
        s_train, s_val = partition(dataset,frac)
        tree0 = fun.buildTree(s_train,data.attributes)
        ptrees = fun.allPruned(tree0)
        score0 = fun.check(tree0, s_val)
        p_score = 0
        top_tree = tree0
        for pt in ptrees:
            p_score = fun.check(pt,s_val)
            if p_score > score0:
                score0 = p_score
                top_tree = pt
        p_error.append(1-fun.check(top_tree,testing))
    return p_error
"prune x times"
monk1_err=[]
monk3_err=[]
for count in range(1000):
    monk1_err.append(pruning(data.monk1,data.monk1test))
    monk3_err.append(pruning(data.monk3,data.monk3test))
print(np.mean(monk1_err,axis=0))
print(np.mean(monk3_err,axis=0))
print(np.var(monk1_err,axis=0))
print(np.var(monk3_err,axis=0))


"plot"
Mean1=np.mean(monk1_err,axis=0)
Mean3=np.mean(monk3_err,axis=0)
Var1=np.var(monk1_err,axis=0)
Var3=np.var(monk3_err,axis=0)

fracs=[0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
monk1m=plt.plot(fracs,Mean1,'ro-',label='monk1m')
monk3m=plt.plot(fracs,Mean3,'bx-',label='monk3m')
"""monk1v=plt.plot(fracs,Var1,'gv-',label='monk1v')
monk3v=plt.plot(fracs,Var3,'ys-',label='monk3v')"""

plt.axis([0.2,0.9,0,0.3])
plt.xlabel('Fractions')
plt.ylabel('Error Variance')
plt.title('Pruning outcome')
plt.legend(loc='upper right')
plt.show()










import numpy as np
import matplotlib.pyplot as plt


#### Digit; Training fs vs non-training fs
b = "0.10025	0.53775	0.50025	0.47	0.43175	0.40675	0.374	0.349	0.339	0.31675	0.30725	0.29625	0.27775	0.2665	0.244	0.2285	0.221	0.20225	0.1925	0.166	0.13925	0.10575	0.106	0.10625	0.10375	0.11075	0.53575	0.47375	0.43625	0.407	0.38075	0.35825	0.339	0.316	0.2935	0.2845	0.27175	0.2525	0.24175	0.23	0.2105	0.19975	0.19075	0.17925	0.1565	0.133	0.10125	0.1005	0.10275	0.104	0.10225"
digit_retrain_fs = [float(i) for i in b.split('\t')]
c = "0.10025	0.53725	0.477	0.43425	0.4155	0.39175	0.37325	0.34675	0.3345	0.32125	0.3085	0.2895	0.28225	0.27725	0.2595	0.25325	0.23775	0.2315	0.20325	0.1945	0.17175	0.1485	0.152	0.14975	0.14125	0.13625	0.5335	0.4815	0.43975	0.421	0.39325	0.36175	0.33975	0.31475	0.299	0.291	0.27225	0.25825	0.2615	0.248	0.239	0.228	0.215	0.208	0.187	0.159	0.1425	0.14275	0.14325	0.13975	0.13925"
digit_no_retrain_fs = [float(i) for i in c.split('\t')]

iters = range(20,71)
fig,ax = plt.subplots()
ax.plot(iters, digit_no_retrain_fs, label='without retraing fs')
ax.plot(iters, digit_retrain_fs, label='retrain fs')
ax.legend()
ax.set_xlabel('Total number of data points')
ax.set_ylabel('Error rate')
ax.set_title('MNIST Data')
default_size = fig.get_size_inches()
size_mult = 1.7
ax.set_ylim((0.0, ax.get_ylim()[1]))
fig.set_size_inches(default_size[0]*size_mult,default_size[1]*size_mult)
plt.show()


#### Newsgroup 10k users
d = "0.0771	0.524	0.46725	0.4235	0.36325	0.3325	0.277	0.24775	0.22275	0.209	0.19875	0.1965	0.19175	0.18625	0.18475	0.17525	0.175	0.1655	0.1665	0.1605	0.1555	0.14525	0.14525	0.14275	0.143	0.13925	0.13875	0.1415	0.1445	0.148	0.14625"
newsgroup_no_retrain_fs = [float(i) for i in d.split('\t')]
e = "0.0771	0.52425	0.4665	0.41425	0.354	0.319	0.264	0.22825	0.204	0.1925	0.1765	0.17	0.16125	0.15375	0.15475	0.1505	0.14875	0.14	0.139	0.1385	0.137	0.13375	0.131	0.1275	0.1245	0.124	0.12225	0.119	0.11975	0.118	0.116"""
newsgroup_retrain_fs = [float(i) for i in e.split('\t')]
iters = range(20,51)
fig,ax = plt.subplots()
ax.plot(iters, newsgroup_no_retrain_fs, label='without retraing fs')
ax.plot(iters, newsgroup_retrain_fs, label='retrain fs')
ax.legend()
ax.set_xlabel('Total number of data points')
ax.set_ylabel('Error rate')
ax.set_title('Newsgroup Data -- 10k Users')
default_size = fig.get_size_inches()
size_mult = 1.7
ax.set_ylim((0.0, ax.get_ylim()[1]))
fig.set_size_inches(default_size[0]*size_mult,default_size[1]*size_mult)
plt.show()


#### Newsgroup 200 users
m = "0.0885	0.52625	0.48625	0.469	0.455	0.44675	0.44225	0.417	0.41125	0.407	0.399	0.38675	0.381	0.37975	0.36775	0.36175	0.352	0.32475	0.322	0.3115	0.29925	0.292	0.28175	0.2705	0.26675	0.258	0.26225	0.24875	0.26775	0.28575	0.289"
newsgroup_no_retrain_fs_200 = [float(i) for i in m.split('\t')]
n = "0.0885	0.50975	0.4375	0.36625	0.336	0.30675	0.29975	0.264	0.24025	0.235	0.22525	0.21825	0.21075	0.1965	0.19275	0.183	0.17425	0.1655	0.15525	0.15275	0.149	0.14025	0.13475	0.12725	0.129	0.12475	0.1195	0.11825	0.11525	0.11275	0.11025"
newsgroup_retrain_fs_200 = [float(i) for i in n.split('\t')]
iters = range(20,51)
fig,ax = plt.subplots()
ax.plot(iters, newsgroup_no_retrain_fs_200, label='without retraing fs')
ax.plot(iters, newsgroup_retrain_fs_200, label='retrain fs')
ax.legend()
ax.set_xlabel('Total number of data points')
ax.set_ylabel('Error rate')
ax.set_title('Newsgroup Data -- 200 Users')
default_size = fig.get_size_inches()
size_mult = 1.7
ax.set_ylim((0.0, ax.get_ylim()[1]))
fig.set_size_inches(default_size[0]*size_mult,default_size[1]*size_mult)
plt.show()


#### Cold start
f = "0.35275	0.2915	0.2585	0.2415	0.249	0.228	0.21275	0.1995	0.19025	0.18475	0.1845	0.17425	0.15925	0.159	0.156	0.15025	0.1475	0.13725	0.13975	0.1405	0.141	0.133	0.1325	0.1275	0.1245	0.1235	0.12075	0.1225"
oracle_mtl_errors = [float(i) for i in f.split('\t')]
g = "0.356	0.31	0.24475	0.232	0.22075	0.2065	0.1995	0.173	0.1825	0.1865	0.17575	0.16525	0.1545	0.1505	0.156	0.15975	0.15725	0.1405	0.14925	0.1395	0.14825	0.14225	0.1405	0.14425	0.1355	0.1365	0.13625	0.13175"
svm_feature_errors = [float(i) for i in g.split('\t')]
h = "0.31725	0.286	0.24275	0.1975	0.1655	0.13875	0.1495	0.13725	0.14925	0.13	0.13375	0.1165	0.1195	0.137	0.12725	0.12925	0.1215	0.1265	0.119	0.11975	0.12075	0.109	0.1195	0.117	0.11875	0.1125	0.11425	0.1145"
svm_feature_errors1 = [float(i) for i in h.split('\t')]
l = "0.36225	0.382	0.35275	0.359	0.33575	0.3565	0.34375	0.33675	0.3365	0.32	0.319	0.329	0.31725	0.317	0.319	0.298	0.30325	0.303	0.293	0.30375	0.289	0.2965	0.29325	0.2825	0.28075	0.28325	0.284	0.28475"
d_feature_errors = [float(i) for i in l.split('\t')]

iters = range(2,30)
fig,ax = plt.subplots()
ax.plot(iters, oracle_mtl_errors, label='oracle')
ax.plot(iters, svm_feature_errors, label = 'svm-l2')
ax.plot(iters, svm_feature_errors1, label = 'svm-l1')
ax.plot(iters, d_feature_errors, label = 'non-sharing')
ax.set_xlabel('Size of Data Points for New User')
ax.set_ylabel('Error Rate')
ax.set_title('Cold Start')
ax.legend()
default_size = fig.get_size_inches()
size_mult = 1.7
ax.set_ylim((0.0, ax.get_ylim()[1]))
fig.set_size_inches(default_size[0]*size_mult,default_size[1]*size_mult)
plt.show()


#### Digit; total concept drift
u = "0.1095625	0.1095625	0.108	0.1075625	0.10675	0.1061875	0.1071875	0.105625	0.1055	0.104125	0.1024375	0.1023125	0.1018125	0.1015625	0.102875	0.10325	0.103375	0.1021875	0.1019375	0.100375	0.0988125	0.5461	0.50355	0.4755	0.44955	0.426675	0.41245	0.398025	0.38315	0.37295	0.3612	0.351875	0.3418	0.33385	0.325825	0.320975	0.31345	0.30565	0.2984	0.291825	0.286375	0.279525	0.275675	0.271075	0.267225	0.261425	0.256775	0.253	0.24795	0.2448	0.24025	0.2369	0.2334	0.22885	0.22655	0.22315	0.22075	0.217525	0.215175	0.213175	0.2102	0.2078	0.20545	0.203625	0.202075	0.1996	0.19755	0.195975	0.193625	0.19165	0.1902	0.188775	0.186625	0.18485	0.183975	0.1825	0.181175	0.1794	0.177775	0.176425	0.174775	0.174075	0.1725	0.172025	0.170875	0.169925	0.169275	0.167875	0.166825	0.1655	0.164425	0.163625	0.16245	0.1613	0.160025	0.15925	0.158075	0.157325	0.15665	0.155625	0.1552	0.154175	0.1529	0.15205	0.151025	0.15065	0.14935	0.1488	0.1483	0.147975	0.147025	0.145725	0.14535	0.144625	0.14435	0.1439	0.143225	0.142325	0.142	0.141775	0.141375"
digit_train_all = [float(i) for i in u.split('\t')]
v = "0.1095625	0.1095625	0.1111875	0.1139375	0.1138125	0.1149375	0.11725	0.1160625	0.1185	0.123125	0.1200625	0.121375	0.12275	0.123125	0.1255	0.1269375	0.127375	0.1278125	0.1295	0.1290625	0.13025	0.543625	0.48275	0.446275	0.408175	0.3901	0.3719	0.353	0.331975	0.314075	0.300575	0.29595	0.281375	0.267475	0.25885	0.245725	0.233075	0.222475	0.20895	0.18925	0.1624	0.1297	0.13425	0.13165	0.1312	0.13255	0.1339	0.132375	0.131575	0.13545	0.1335	0.13305	0.133225	0.1342	0.133375	0.13215	0.130725	0.127625	0.1301	0.13	0.129175	0.12965	0.12825	0.128575	0.126425	0.1274	0.127625	0.1293	0.130825	0.1296	0.13025	0.13405	0.13315	0.13275	0.133075	0.13055	0.132075	0.13155	0.131225	0.130425	0.133475	0.131575	0.126975	0.12655	0.1264	0.127525	0.128425	0.12835	0.12575	0.1272	0.127325	0.127825	0.126525	0.1296	0.128325	0.129475	0.127525	0.1267	0.1252	0.123025	0.12175	0.122975	0.12615	0.128525	0.1256	0.127	0.130425	0.13025	0.129725	0.129975	0.128725	0.127375	0.1282	0.129625	0.128475	0.1273	0.12985	0.1302	0.12885	0.13005	0.12965"
digit_retrain_new = [float(i) for i in v.split('\t')]
w = "0.1095625	0.1095625	0.1095625	0.1095625	0.1095625	0.1095625	0.1095625	0.1095625	0.1095625	0.1095625	0.1095625	0.1095625	0.1095625	0.1095625	0.1095625	0.1095625	0.1095625	0.1095625	0.1095625	0.1095625	0.1095625	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955	0.53955"
digit_no_retrain = [float(i) for i in w.split('\t')]

iters = range(20, 20+len(digit_train_all))
fig,ax = plt.subplots()
ax.plot(iters, digit_train_all, label='retrain-all-data')
ax.plot(iters, digit_retrain_new, label = 'retrain-latest-data')
ax.plot(iters, digit_no_retrain, label = 'no-retrain')
ax.set_xlabel('Size of Total Data')
ax.set_ylabel('Error Rate')
ax.set_title('Total Concept Drift - MNIST Data')
ax.legend(loc=4)
default_size = fig.get_size_inches()
size_mult = 1.7
ax.set_ylim((0.0, ax.get_ylim()[1]))
fig.set_size_inches(default_size[0]*size_mult,default_size[1]*size_mult)
plt.show()


#### digit, partial concept drift
o = "0.1095625	0.1095625	0.1070625	0.108	0.1065625	0.106125	0.1050625	0.1045	0.1038125	0.102875	0.10425	0.102875	0.103	0.1016875	0.1006875	0.1005625	0.1025	0.1006875	0.10175	0.100625	0.0995625	0.301175	0.2898	0.281625	0.27815	0.273125	0.270275	0.264225	0.25935	0.257075	0.2542	0.250725	0.246975	0.2433	0.23905	0.2359	0.23365	0.231275	0.2286	0.226375	0.223975	0.220275	0.21685	0.21485	0.212	0.210875	0.207725	0.206	0.204	0.202175	0.1999	0.1983	0.196175	0.1955	0.19525	0.1936	0.190525	0.19025	0.188325	0.187525	0.185425	0.18435	0.182525	0.180925	0.180525	0.179025	0.178225	0.1772	0.176275	0.175875	0.175275	0.1745	0.173325	0.172825	0.172175	0.170925	0.170175	0.1695	0.16865	0.168725	0.16675	0.166	0.1652	0.1653	0.16495	0.164775	0.163975	0.16315	0.162325	0.161575	0.160625	0.1602	0.16015	0.1596	0.15895	0.15875	0.158175	0.157725	0.1576	0.1577	0.156725	0.1565	0.155975	0.1555	0.15495	0.15485	0.154675	0.154125	0.153725	0.153275	0.15285	0.152475	0.15255	0.151725	0.1517	0.151625	0.151125	0.151	0.15055	0.150475	0.1502"
digit_train_all = [float(i) for i in o.split('\t')]
p = "0.1095625	0.1095625	0.1093125	0.1113125	0.1124375	0.11425	0.1140625	0.1146875	0.11775	0.1190625	0.1210625	0.121	0.120875	0.123	0.122	0.12225	0.1245	0.1246875	0.12575	0.125375	0.12525	0.311675	0.2997	0.289375	0.28445	0.278775	0.2768	0.268	0.264225	0.258525	0.25405	0.2504	0.2454	0.2436	0.234975	0.2312	0.2298	0.229325	0.23085	0.228775	0.22825	0.2253	0.2227	0.22555	0.2216	0.22045	0.21785	0.2165	0.2183	0.220425	0.21995	0.220075	0.2213	0.22005	0.218375	0.218325	0.215025	0.218875	0.2187	0.217525	0.215425	0.2181	0.218775	0.22245	0.22395	0.2235	0.224825	0.221975	0.221525	0.22085	0.2255	0.2255	0.224725	0.2239	0.222025	0.22195	0.224875	0.224925	0.225525	0.2268	0.2267	0.2234	0.22225	0.22095	0.223325	0.220075	0.22195	0.223525	0.22425	0.221075	0.217975	0.218475	0.22135	0.220725	0.221825	0.221425	0.222325	0.2198	0.220575	0.221575	0.222	0.2232	0.224375	0.2241	0.2238	0.225075	0.227825	0.226125	0.2259	0.227325	0.2258	0.2275	0.225975	0.22555	0.228725	0.22935	0.2277	0.2277	0.228625	0.23025	0.229375"
digit_retrain_new = [float(i) for i in p.split('\t')]
q = "0.1095625	0.1095625	0.1095625	0.1095625	0.1095625	0.1095625	0.1095625	0.1095625	0.1095625	0.1095625	0.1095625	0.1095625	0.1095625	0.1095625	0.1095625	0.1095625	0.1095625	0.1095625	0.1095625	0.1095625	0.1095625	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675	0.305675"
digit_no_retrain = [float(i) for i in q.split('\t')]

iters = range(20, 20+len(digit_train_all))
fig,ax = plt.subplots()
ax.plot(iters, digit_train_all, label='retrain-all-data')
ax.plot(iters, digit_retrain_new, label = 'retrain-latest-data')
ax.plot(iters, digit_no_retrain, label = 'no-retrain')
ax.set_xlabel('Size of Total Data')
ax.set_ylabel('Error Rate')
ax.set_title('Partial Concept Drift - MNIST Data')
ax.legend(loc=4)
default_size = fig.get_size_inches()
size_mult = 1.7
ax.set_ylim((0.0, ax.get_ylim()[1]))
fig.set_size_inches(default_size[0]*size_mult,default_size[1]*size_mult)
plt.show()


#### cache miss vs accuracy
aa = "0.0001520086862	0.0001628664495	0.0003583061889	0.0002823018458	0.0005320304017	0.0005211726384	0.0007383279045	0.0009771986971	0.001465798046	0.0008903365907	0.001704668838	0.002953311618	0.002225841477	0.003105320304	0.002160694897	0.002214983713	0.007198697068	0.005157437568	0.005027144408	0.007459283388	0.008870792617	0.01465798046	0.01061889251	0.0161237785	0.012247557	0.01438653637	0.0160369164	0.01445168295	0.01752442997	0.0296742671	0.02281216069	0.02043431053	0.02314875136	0.02563517915	0.04859934853	0.04107491857	0.03116178067	0.04377850163	0.06319218241	0.1004234528	0.05250814332	0.05407166124	0.06421281216	0.05442996743	0.06479913138	0.06853420195	0.08271444083	0.07996742671	0.1147014115	0.1075461455	0.07007600434	0.1141476656	0.1052225841	0.116742671	0.1542996743	0.1147448426	0.1450054289	0.1788165038	0.1323344191	0.1322584148	0.1426384365	0.217383279	0.161422367	0.1739956569	0.2803040174	0.2615743757	0.2504343105	0.2388490771	0.2100977199	0.2841042345	0.2767752443	0.2730184582	0.3385884908	0.3375787188	0.3973507058	0.2778501629	0.2872204126	0.3397394137	0.3158306189	0.344679696	0.3245276873	0.314723127	0.3596525516	0.4016395223	0.4084473398	0.4342453855	0.4055700326	0.3916286645	0.3775461455	0.4394571118	0.4583604777	0.5096959826	0.4248317047	0.4303148751	0.4153420195	0.5999022801	0.4406188925	0.525548317	0.5664603692	0.002377850163	0.001910966341	0.001900108578	0.001465798046	0.003507057546	0.003550488599	0.003908794788	0.004397394137	0.007068403909	0.007535287731	0.006970684039	0.00552660152	0.008414766558	0.008501628665	0.007133550489	0.01491856678	0.008165038002	0.01922909881	0.01605863192	0.01731813246	0.0282844734	0.02713355049	0.01737242128	0.02194353963	0.02528773073	0.05348534202	0.03529858849	0.02444082519	0.03277958741	0.04034744843	0.06993485342	0.04776330076	0.09682953312	0.06636264929	0.07457111835	0.0665689468	0.1107709012	0.1108577633	0.103257329	0.06203040174	0.09242128122	0.1118241042	0.1009446254	0.1151791531	0.1067643865	0.1733984799	0.1358957655	0.09833876222	0.1226492942	0.1476655809	0.1291856678	0.2217480999	0.1856243214	0.2417480999	0.2067969598	0.2821932682	0.1785993485	0.1805211726	0.3472855592	0.2939196526	0.2227904452	0.3417589577	0.2139522258	0.2404560261	0.2550054289	0.3104668838	0.3192616721	0.2950488599	0.3862214984	0.3328230185	0.3578501629	0.4256568947	0.3887730727	0.4752117264	0.4584690554	0.347937025	0.4083604777	0.4372747014	0.3805428882	0.4360694897	0.4132247557	0.4314549403	0.4199239957	0.5031161781	0.5108686211	0.6078718784	0.4835613464	0.5024972856	0.6072529859	0.4300977199	0.51082519	0.5728121607	0.579359392	0.5750380022	0.6228230185	0.5735179153	0.5518023887	0.6218675353	0.6322692725	0.5689359392	0.6068295331	0.5586210641	0.6055917481	0.6186210641	0.6461563518	0.5787296417	0.6599348534	0.7030076004	0.7342236699	0.6437459283	0.7371986971	0.7105211726	0.7141476656	0.6813897937	0.6755917481	0.7578718784	0.7035613464	0.751422367	0.7053311618	0.6660152009	0.8371335505	0.7305320304	0.6851791531	0.8404343105	0.742432139	0.7595656895	0.71082519	0.8647774159	0.802432139	0.80165038	0.7953528773	0.8084690554	0.7818241042	0.8343973941	0.883029316	0.8504017372	0.7941042345	0.9032790445	0.8331596091	0.8130401737	0.8281758958	0.8241585233	0.806286645	0.8874375679	0.8309880565	0.8363952226	0.8522801303	0.8799782845	0.8848859935	0.8878718784	0.854907709	0.855548317	0.8902605863	0.8661780673	0.9214549403	0.8948099891	0.9133441911	0.9219978284	0.8971444083	0.8801737242	0.8790119435	0.8913137894	0.9360260586	0.9272855592	0.9169923996	0.9185450597	0.9286427796	0.9160152009	0.9325515744	0.9221064061	0.9176112921	0.9181976113	0.9463517915	0.944495114	0.9397176982	0.9363083605	0.9460260586	0.9390553746	0.9351682953	0.9587187839	0.9530184582	0.9264603692	0.9568186754	0.9420846906	0.9497176982	0.9623452769	0.9496851249	0.9474484256	0.9666232356	0.9535613464	0.9584039088	0.9562975027	0.9211834962	0.9687730727	0.9542888165	0.965092291	0.9529858849	0.9730618893	0.9650705755	0.9415092291	0.9618566775	0.9674484256	0.9643431053	0.9712595005	0.9659826276	0.9570575461	0.967752443	0.9775787188	0.9614549403	0.9596525516	0.9703148751	0.9609771987	0.9763843648	0.9583604777	0.9660043431	0.966742671	0.9760912052	0.9635939197	0.9790662324	0.9759500543	0.9783387622	0.971009772	0.9675027144	0.9752008686	0.9760912052	0.976102063	0.9825298588	0.9751574376	0.9860369164	0.9833659066	0.9814766558	0.9728230185	0.9830184582	0.983854506	0.9816178067	0.9829641694	0.9760260586	0.9786970684	0.9866123779	0.9837893594	0.9850488599	0.983854506	0.9799239957	0.967155266	0.9770141151	0.9856134636	0.9874592834	0.9791530945	0.9906948969	0.9781216069	0.9870466884	0.9834636265	0.9857003257	0.9912052117	0.9891639522	0.9878501629	0.9878501629	0.9845385451	0.9823995657	0.9852225841	0.984495114	0.9879587405	0.9879261672	0.9847122693	0.975689468	0.9860694897	0.9915200869	0.9909554832	0.9911834962	0.9854614549	0.9876438654	0.9873181325	0.9879695983	0.9889033659	0.9882410423	0.9878827362	0.9886536374	0.9926492942	0.9911726384	0.9854505972	0.9908903366	0.9935287731	0.9930184582	0.9921824104	0.9910749186	0.9909011944	0.9888707926	0.9922692725	0.9904017372	0.9891856678	0.9926927253	0.9944625407	0.9923127036	0.9882084691	0.9924864278	0.9913789359	0.9927578719	0.9951900109	0.9940282302	0.993897937	0.99247557	0.9923669924	0.9938110749	0.990184582	0.9950162866	0.9942128122	0.9933441911	0.9933876222	0.9952334419	0.9930618893	0.9941585233	0.9942453855	0.9960043431	0.993897937	0.9900542888	0.9945819761	0.9918458198	0.9961780673	0.9931921824	0.9943431053	0.9939739414	0.9956677524	0.9930184582	0.9960803474	0.9938436482	0.9954071661	0.9954723127	0.9966123779	0.9936264929	0.9960586319	0.9962214984	0.9966340934	0.9949619978	0.9944733985	0.9960152009	0.9955700326	0.9964495114	0.9965255157	0.9970684039	0.9968621064	0.9974049946	0.9978175896	0.9961997828	0.9966449511	0.9982736156	0.9957654723	0.9970358306	0.9950814332	0.995504886	0.9966123779	0.9964820847	0.994907709	0.9976330076	0.9967643865	0.9962540717	0.9972638436	0.9971986971	0.9947991314	0.9954505972	0.997383279	0.9973724213	0.9968078176	0.997567861	0.9963843648	0.9971769815	0.9963517915	0.9969381107	0.9980238871	0.9964712269	0.997339848	0.9971769815	0.9970792617	0.9980564604	0.995092291	0.995689468	0.9980347448	0.9983279045	0.9978284473	0.9972529859	0.9967969598	0.9978393051	0.9977415852	0.9981867535	0.9986210641	0.9976872964	0.9986102063	0.9976655809	0.9986102063	0.9988599349	0.9979478827"
cache_miss_rate = [str(i) for i in aa.split('\t')]
bb = "0.3724	0.3784	0.3885	0.3797	0.4055	0.38	0.3731	0.3919	0.3811	0.3819	0.3725	0.3841	0.3821	0.3833	0.3634	0.3729	0.3822	0.3613	0.3905	0.3766	0.375	0.3673	0.3754	0.3554	0.3746	0.388	0.348	0.3678	0.3611	0.3577	0.3498	0.3568	0.3511	0.3654	0.3556	0.3292	0.3552	0.3202	0.3238	0.3239	0.3509	0.3434	0.3526	0.3334	0.3387	0.3409	0.3416	0.3445	0.332	0.3209	0.3407	0.3188	0.3232	0.3336	0.3166	0.3257	0.3248	0.3088	0.3311	0.332	0.3401	0.3164	0.315	0.3232	0.317	0.3161	0.3249	0.3084	0.3201	0.3116	0.3009	0.3001	0.3061	0.32	0.3079	0.3067	0.3155	0.306	0.3168	0.3005	0.3173	0.3112	0.3	0.3186	0.3135	0.3058	0.2902	0.3058	0.3072	0.2991	0.3059	0.3116	0.3071	0.3092	0.2998	0.3161	0.3162	0.308	0.3043	0.3726	0.3995	0.3981	0.3923	0.4013	0.3664	0.3583	0.375	0.3661	0.3786	0.361	0.3925	0.36	0.37	0.3605	0.3693	0.3791	0.3648	0.3561	0.3689	0.3608	0.3756	0.3904	0.3558	0.3504	0.3302	0.3614	0.3562	0.3559	0.3474	0.3393	0.3544	0.3588	0.3314	0.3378	0.3443	0.3284	0.3295	0.3449	0.3525	0.3505	0.3091	0.326	0.3306	0.3169	0.3254	0.3251	0.3302	0.3499	0.3208	0.3453	0.3147	0.3166	0.325	0.3233	0.3069	0.3259	0.3133	0.3043	0.3181	0.3195	0.3043	0.3187	0.3181	0.3089	0.3139	0.3135	0.3235	0.3064	0.3015	0.3128	0.3041	0.2974	0.3024	0.2929	0.3004	0.308	0.3012	0.3159	0.3057	0.3093	0.3051	0.3128	0.3029	0.2971	0.2979	0.293	0.2998	0.3103	0.3168	0.3054	0.3151	0.3038	0.3103	0.2978	0.2946	0.3077	0.3011	0.3038	0.2916	0.3029	0.3044	0.3071	0.2983	0.3093	0.3001	0.3096	0.3101	0.3133	0.3046	0.3056	0.3044	0.3051	0.2977	0.2929	0.3077	0.2998	0.3128	0.3045	0.3062	0.3142	0.3102	0.3059	0.3165	0.3116	0.3127	0.3105	0.3099	0.3175	0.3094	0.3194	0.3181	0.3127	0.3183	0.3133	0.3164	0.3111	0.3179	0.3043	0.3175	0.3176	0.3131	0.313	0.3179	0.3198	0.3116	0.307	0.3226	0.3185	0.3151	0.3082	0.3248	0.3193	0.3106	0.3289	0.3169	0.3208	0.321	0.3197	0.3272	0.3204	0.3104	0.3238	0.3158	0.3246	0.3187	0.3272	0.3173	0.3211	0.3141	0.3205	0.3163	0.3201	0.3225	0.3181	0.3232	0.329	0.3232	0.3259	0.3251	0.3297	0.3165	0.3253	0.3194	0.3233	0.3213	0.3218	0.3273	0.3292	0.3242	0.329	0.324	0.3245	0.329	0.3232	0.3246	0.324	0.3206	0.3248	0.3171	0.3286	0.3239	0.3264	0.3288	0.3322	0.3278	0.3322	0.3285	0.3319	0.3282	0.3319	0.3242	0.333	0.3238	0.3266	0.3261	0.3282	0.324	0.3284	0.3314	0.3297	0.3289	0.3292	0.3336	0.3253	0.3286	0.3286	0.3323	0.3308	0.3294	0.3317	0.3318	0.3306	0.3294	0.3287	0.3338	0.3333	0.3303	0.3318	0.3327	0.333	0.3323	0.3293	0.3278	0.3288	0.3327	0.3302	0.3274	0.334	0.333	0.3329	0.3307	0.3317	0.331	0.3336	0.3332	0.3334	0.3315	0.3303	0.3287	0.3286	0.3281	0.3311	0.3302	0.3316	0.331	0.3316	0.3332	0.3328	0.3321	0.328	0.3337	0.3336	0.334	0.3342	0.3329	0.3329	0.3328	0.3323	0.3292	0.3309	0.3339	0.3337	0.3341	0.3326	0.3309	0.3336	0.3302	0.3339	0.3336	0.3331	0.3349	0.3336	0.3349	0.334	0.3328	0.3343	0.3339	0.3346	0.3323	0.3336	0.3337	0.3313	0.327	0.3301	0.3337	0.3326	0.334	0.333	0.3333	0.3341	0.3342	0.3341	0.3339	0.3341	0.3318	0.3342	0.3315	0.3342	0.3344	0.3346	0.3338	0.3322	0.3341	0.3346	0.3344	0.3342	0.3342	0.3325	0.3333	0.3342	0.3339	0.3345	0.3346	0.334	0.334	0.3343	0.3343	0.3332	0.3342	0.3338	0.3341	0.3341	0.3333	0.3343	0.3344	0.3333	0.3341	0.3346	0.3347	0.3343	0.3349	0.3346	0.3343	0.3348	0.3342	0.3346	0.3318	0.3345	0.3346	0.3345	0.3343	0.3334	0.3345	0.3338	0.3344	0.3334	0.3347	0.3316	0.3341	0.3333	0.3346	0.3325	0.3346	0.3345	0.3342	0.3348	0.3349	0.3334	0.3341	0.3349	0.3343	0.3343	0.3344	0.3346	0.3348	0.334	0.3336	0.3343	0.3348"
error_list = [str(i) for i in bb.split('\t')]

fig,ax = plt.subplots()
ax.scatter(cache_miss_rate, error_list)
#ax.plot([0.0,1.0],[error, error],'--', label="non-cache")
ax.set_xlabel('cache miss rate')
ax.set_ylabel('error rate')
#ax.set_ylim((0,ax.get_ylim()[1]))
ax.legend()
default_size = fig.get_size_inches()
size_mult = 1.1
fig.set_size_inches(default_size[0]*size_mult,default_size[1]*size_mult)
plt.show()









































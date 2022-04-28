tra:
	/usr/local/cuda/bin/nvcc tra.cu `pkg-config opencv --cflags --libs` tra.cpp -o tra 
	./tra f1.jpg 2.0
ari:
	/usr/local/cuda/bin/nvcc ari.cu `pkg-config opencv --cflags --libs` ari.cpp -o ari 
	./ari f1.jpg f2.jpg 3
gau1:
	/usr/local/cuda/bin/nvcc gau.cu `pkg-config opencv --cflags --libs` gau.cpp -o gau 
	./gau sapo.jpg g 15 1
lap:
	/usr/local/cuda/bin/nvcc gau.cu `pkg-config opencv --cflags --libs` gau.cpp -o gau 
	./gau sapo.jpg l
sep:
	/usr/local/cuda/bin/nvcc sep.cu `pkg-config opencv --cflags --libs` sep.cpp -o sep 
	./sep sapo.jpg 7 1
knn:
	/usr/local/cuda/bin/nvcc knn.cu `pkg-config opencv --cflags --libs` knn.cpp -o knn 
	./knn sapo.jpg 7 50
col:
	/usr/local/cuda/bin/nvcc col.cu `pkg-config opencv --cflags --libs` col.cpp -o col 
	./col sapo.jpg 180
gauSM:
	/usr/local/cuda/bin/nvcc gauSM.cu `pkg-config opencv --cflags --libs` gau.cpp -o gauSM 
	./gauSM sapo.jpg g 15 1
lapSM:
	/usr/local/cuda/bin/nvcc gauSM.cu `pkg-config opencv --cflags --libs` gau.cpp -o gauSM 
	./gauSM sapo.jpg l
sepSM:
	/usr/local/cuda/bin/nvcc sepSM.cu `pkg-config opencv --cflags --libs` sep.cpp -o sepSM 
	./sepSM sapo.jpg 15 1
knnSM:
	/usr/local/cuda/bin/nvcc knnSM.cu `pkg-config opencv --cflags --libs` knn.cpp -o knnSM 
	./knnSM sapo.jpg 7 50
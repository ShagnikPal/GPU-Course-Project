main:
	nvcc -o main.out main.cu
	./main.out Sparse_Graph.txt
	./main.out Random_Dense_Graph.txt
	./main.out Fully_Connected_Graph.txt

InputGen:
	rm *.txt
	python gen_graph.py -gpn -n 60001 >> Sparse_Graph.txt
	python gen_graph.py -grnm -n 1001 -m 20000 >> Random_Dense_Graph.txt
	python gen_graph.py -gkn -n 361 >> Fully_Connected_Graph.txt

clean:
	rm *.out
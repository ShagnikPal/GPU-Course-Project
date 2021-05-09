#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <stdlib.h>
#include <stdio.h>
#include <limits.h>
#include <math.h>
#include <time.h>
#include<bits/stdc++.h>
using namespace std;

#define shamt 32
#define vertex_bits	22	//Increse this for more vertexes. But then may need to reduce weights. If wt. also large, may need to change to long
#define INVALID 1<<23	//Since no vertice can have this value. Max they can have is 2^23-1

unsigned int VertexSize;    //Number of vertices at each iteration
unsigned int EdgeSize;    //Number of vertices at each iteration
unsigned int VertexSize_cpu	;	//Constant: number of vertices at start of program
unsigned int EdgeSize_cpu;		//Constant: number of edges at start of program
unsigned int *cpu_vertex;	//Contains the vertex list, having start offset of the edgesegment in the edgelist
unsigned int *cpu_edge_dst;	//dst vertex for each edge
unsigned int *cpu_edge_weight;	//wt of each edge
unsigned int *gpu_vertex, *gpu_edge_dst, *gpu_edge_weight; //CSR arrays on GPU 
unsigned int *gpu_edge_weight_temp;	//Used to prevent race condition. Contains a copy
unsigned int *gpu_edge_dst_temp;	////Used to prevent race condition. Contains a copy
unsigned int *gpu_edge_WV;	// Has LSB 22 bits as destination vertex id and MSB 10 bits as edge weight 
unsigned int *gpu_edge_mapping;	//Mapping to index to |E| sized original edge array for a particular edge
unsigned int *gpu_edge_mapping_temp;	//Used to prevent race condition
unsigned int *gpu_min_edge_index;	//Stores the min edge index which is the min found in Findmin
unsigned int *gpu_min_edge;	//Output of min edge found for every vertex. Output stored at end of every edge segment
unsigned int *gpu_successor;	//Contains successor for every vertex
unsigned int *gpu_successor_temp;	//Copy of successor, read only, for sync purposes
unsigned int *gpu_vertex_list;	//Has |E| (no of edges in that iter) size src vertex list for each edge
unsigned int *cpu_vertex_list;	//CPU version of above
unsigned int *gpu_OutputMST;	//Marks indexes of edges to be included in final MST
unsigned long *gpu_sorted_vertex;	//Has MSB 32 bits as succ[i] and LSB 32 bits as i
unsigned int *gpu_super_vertex_list;	//New vertex list of size after they are grouped under a supervertex. Contains SuperVertex
unsigned int *gpu_parent_supervertex;	//Tells which supervertex each original vertex now belongs to
unsigned long *gpu_new_concat_vertices;	//Contains the concatenated 64 bit (dst_supervertex,src_supervertex), 32 bit each
unsigned int *gpu_new_EdgeSize;		//Contains the new EdgeSize after nullifying edges among same supervertices
unsigned int *gpu_new_VertexSize;	//Contains the new VertexSize after replacing vertices with their supervertices
int flag;	//To check if Boruvka MST has been completed

//-------------------------------THE KERNELS-------------------------------------------------

//This concatenates dst vertex and the weight of edge. MSB has weight so that it is given priority while finding min among edges.
__global__ void ConcatenateWV(unsigned int *gpu_edge_WV, unsigned int *gpu_edge_weight, unsigned int *gpu_edge_dst, unsigned int EdgeSize){
    unsigned int id=blockIdx.x*blockDim.x + threadIdx.x;
	if(id<EdgeSize){
		unsigned int temp = gpu_edge_weight[id]<<vertex_bits;
		gpu_edge_WV[id]= temp | gpu_edge_dst[id];
	}
}

//Find Minimum Weighted Edge, for a given src vertex, among all edges connected to it
__global__ void FindMin(unsigned int *gpu_min_edge, unsigned int *gpu_edge_WV,unsigned int *gpu_vertex, unsigned int *gpu_min_edge_index, unsigned int *gpu_edge_mapping, unsigned int * gpu_successor,unsigned int VertexSize, unsigned int EdgeSize){
    unsigned int id=blockIdx.x*blockDim.x + threadIdx.x;
	if(id<VertexSize){
		unsigned int start=gpu_vertex[id];
		unsigned int end;
		if(id<VertexSize-1)
			end = gpu_vertex[id+1]-1;
		else
			end = EdgeSize-1;
		unsigned int min = UINT_MAX;
		unsigned int index=0;
		//Find minimum in list of edges for that src vertex
		for(unsigned int i=start;i<=end;i++){
			if(gpu_edge_WV[i]<min){
				min = gpu_edge_WV[i];
				index=i;	//Store it's index in the edge list
			}
		}
		gpu_min_edge[id]=min;
		gpu_min_edge_index[id]= gpu_edge_mapping[index];
		//Assign Succesor
		unsigned long mask = pow(2.0, vertex_bits)-1;
		gpu_successor[id]= gpu_min_edge[id] & mask;	//Take LSB 22 bits of dst vertex id of as successor
	}
}

//Remove cycles between two vertices. Note that cycles can exist only between 2 vertices
__global__ void RemoveCycles(unsigned int *gpu_successor, unsigned int VertexSize){
    unsigned int id=blockIdx.x*blockDim.x + threadIdx.x;
	if(id<VertexSize){
		unsigned int parent = gpu_successor[id];
		unsigned int grandparent = gpu_successor[parent];
		if(id == grandparent)//Found a Cycle
		{
			//Assign the lower numbered vertice as successor. For a,b if a<b. succ[a]=succ[b]=a
			if(id < parent)
				gpu_successor[id]=id;
			else
				gpu_successor[parent]=parent;
		}
	}
}

//Initialize Array with all 0's. Used this because cudaMemset was not working properly
__global__ void InitiaizeZero(unsigned int *Array, unsigned int len){
    unsigned int id=blockIdx.x*blockDim.x + threadIdx.x;
	if(id<len){
		Array[id]=0;
	}
}

//Mark 1 at boundaries of edge segment for a vertex
__global__ void MarkSegment(unsigned int *gpu_vertex_list, unsigned int *gpu_vertex, unsigned int VertexSize){
    unsigned int id=blockIdx.x*blockDim.x + threadIdx.x;
	if(id<VertexSize){
		if(id>0)	//Skip marking for 1st flag cuz the 1st vertex is zero
			gpu_vertex_list[gpu_vertex[id]]=1;	
	}
}

//Call with 1 thread. Calls thrust to do inclusive scan.
__global__ void InclusiveScan(unsigned int *gpu_vertex_list, unsigned int EdgeSize){
	thrust::inclusive_scan(thrust::device, gpu_vertex_list, gpu_vertex_list + EdgeSize, gpu_vertex_list); 
}

//Mark indexes in original edge list which should be included in final MST
__global__ void OutputMST(unsigned int* gpu_OutputMST,unsigned int* gpu_successor, unsigned int *gpu_min_edge_index, unsigned int VertexSize){
    unsigned int id=blockIdx.x*blockDim.x + threadIdx.x;
	if(id<VertexSize){
		if(id!=gpu_successor[id]){	//Make sure not marking edges twice for pairwise cyclic vertices
			unsigned int v = gpu_min_edge_index[id];
			gpu_OutputMST[v]=1;
		}
	}
}

//Copy src to dst
__global__ void CopyArray(unsigned int *dst, unsigned int *src, unsigned int len){
    unsigned int id=blockIdx.x*blockDim.x + threadIdx.x;
	if(id<len){
		dst[id]=src[id];
	}
}

//Propogate Supervertices. Keep doing, S(u)=S(S(u)) until both are same, which means we reached the representative vertex
//Note that each set of connected edges have a unique supervertex, since there can't be 2 pairwise cycles in same set of connected edges
__global__ void PropogateRepresentatives(unsigned int *gpu_successor, unsigned int *gpu_successor_temp, unsigned int VertexSize){
    unsigned int id=blockIdx.x*blockDim.x + threadIdx.x;
	if(id<VertexSize){
		unsigned int u = id;	//Note that succ_temp is read only
		while(u!=gpu_successor_temp[u]){
			u = gpu_successor_temp[u];
		}
		gpu_successor[id]=u;
	}
}

//Concatenate two 32 bit succ[i] & i into 64 bit (succ[i],i)
__global__ void ConcatenateSucc(unsigned long *gpu_sorted_vertex, unsigned int *gpu_successor, unsigned int VertexSize){
    unsigned int id=blockIdx.x*blockDim.x + threadIdx.x;
	if(id<VertexSize){
		unsigned long temp = gpu_successor[id];
		temp = temp<<shamt;
		gpu_sorted_vertex[id] = temp | id;
	}
}

//Call with 1 thread. Calls thrust to do sort. 
//Sorts the vertices according to (succ[i],i)
__global__ void SortVertices(unsigned long *A, unsigned int n){
	thrust::stable_sort(thrust::device, A, A + n); 
}

//Marks flags when succ[i] is different for consecutive elements. Will be used in inclusive scan
__global__ void MarkSuperVertexFlags(unsigned int *gpu_super_vertex_list, unsigned long *gpu_sorted_vertex, unsigned int VertexSize){
    unsigned int id=blockIdx.x*blockDim.x + threadIdx.x;
	if(id<VertexSize){
		if(id==0){
			gpu_super_vertex_list[id]=0;
		}
		else{
			unsigned long temp1=gpu_sorted_vertex[id]>>shamt;
			unsigned long temp2=gpu_sorted_vertex[id-1]>>shamt;
			if(temp1!=temp2){
				gpu_super_vertex_list[id]=1;
			}
		}
	}
}

//For each vertex, assign its parent supervertex
__global__ void AssignParentSuperVertex(unsigned int *gpu_parent_supervertex, unsigned int *gpu_super_vertex_list, unsigned long *gpu_sorted_vertex, unsigned int VertexSize){
	unsigned int id=blockIdx.x*blockDim.x + threadIdx.x;
	if(id<VertexSize)
	{
		unsigned long mask = pow(2.0, shamt)-1;
		unsigned long temp = gpu_sorted_vertex[id] & mask;
		gpu_parent_supervertex[temp] = gpu_super_vertex_list[id];	//For each vertex at location id in gpu_vertex, it's parent assigned
	}
}

//Remove Internal edges between vertices belonging to the same supervertex
__global__ void RemoveInternalEdges(unsigned int *gpu_edge_dst, unsigned int *gpu_vertex_list, unsigned int *gpu_parent_supervertex, unsigned int EdgeSize){
	unsigned int id=blockIdx.x*blockDim.x + threadIdx.x;
	if(id<EdgeSize){
		//Find the edge u-v in edgelist index id
		unsigned int u = gpu_vertex_list[id];	//src
		unsigned int v = gpu_edge_dst[id];		//dst
		//Find the supervertices of u and v and check if they belong to same supervertex
		unsigned int supervertex_u = gpu_parent_supervertex[u];
		unsigned int supervertex_v = gpu_parent_supervertex[v];
		if(supervertex_u == supervertex_v){
			//If both vertices belong to same supervertex, that edge is invalid
			gpu_edge_dst[id]=INVALID; 
			gpu_vertex_list[id]=INVALID;
		}
	}
}

//gpu_new_concat_vertices contains in the MSB 32 bits-> supervertex. LSB 32 bits -> edge index
//For invalid edges, MSB 32 bits will have INVALID (1<<23). Thus while sorting, it will go at the end
__global__ void MakeNewConcatnatedVertices(unsigned long *gpu_new_concat_vertices, unsigned int *gpu_edge_dst, unsigned int *gpu_vertex_list, unsigned int *gpu_parent_supervertex, unsigned int EdgeSize){
	unsigned int id=blockIdx.x*blockDim.x + threadIdx.x;
	if(id<EdgeSize){
		unsigned long temp;
		unsigned int u,v,supervertex=INVALID;
		u = gpu_vertex_list[id];	//src
		v = gpu_edge_dst[id];		//dst
		if(u!=INVALID && v!=INVALID)	//Check if both src and dst vertice are not nullified, ie, don't belong to same supervertex
			supervertex = gpu_parent_supervertex[u];
		temp = supervertex;
		temp = temp<<shamt;	
		gpu_new_concat_vertices[id] = temp | id ; //Contains (supervertex, edgelist index)
	}
}

//Updates Edgesize after removing nullified edges
//Note that gpu_new_concat_vert has sorted (dst,src) vertices
__global__ void UpdateEdgeSize(unsigned long *gpu_new_concat_vertices, unsigned int *gpu_new_EdgeSize, unsigned int EdgeSize){
	unsigned int id=blockIdx.x*blockDim.x + threadIdx.x;
	if(id<EdgeSize){
		if(id>0){
			unsigned long curr = gpu_new_concat_vertices[id]>>shamt;
			unsigned long prev = gpu_new_concat_vertices[id-1]>>shamt;
			if(curr == INVALID && prev != INVALID)	//First instance of invalid edge. So this is the new edgesize
				*gpu_new_EdgeSize = id; 
		}
		else{//here id=0;
			unsigned long curr = gpu_new_concat_vertices[id]>>shamt;
			if(curr==INVALID){
				*gpu_new_EdgeSize = 0;	//If the first edge itself is invalid, no edges remain to add. MST is done
			}	
		}
	}
}

//Update the edge list, and the mapping of current edges to original edges of edgelist
__global__ void UpdateEdgeList(unsigned long *gpu_new_concat_vertices, unsigned int *gpu_edge_mapping, unsigned int *gpu_edge_mapping_temp, unsigned int *gpu_edge_dst, unsigned int *gpu_edge_dst_temp, unsigned int *gpu_parent_supervertex, unsigned int *gpu_edge_weight, unsigned int *gpu_edge_weight_temp, unsigned int *gpu_vertex_list, unsigned int *gpu_new_VertexSize, unsigned int New_EdgeSize){
	unsigned int id=blockIdx.x*blockDim.x + threadIdx.x;
	if(id<New_EdgeSize){
		unsigned long temp = gpu_new_concat_vertices[id]; 
		unsigned long mask = pow(2.0, shamt)-1;
		unsigned long index = temp & mask;	//Contains index to edgelist which has the corresponding dst vertex
		unsigned long u = temp >> shamt;	//u is the src supervertex
		unsigned int v = gpu_edge_dst_temp[index];	//v has the dst vertex number (non supervertex)
		if(u!=INVALID && v!=INVALID){
			gpu_edge_mapping[id] = gpu_edge_mapping_temp[index];
			gpu_edge_dst[id] = gpu_parent_supervertex[v];
			gpu_edge_weight[id] = gpu_edge_weight_temp[index];
			gpu_vertex_list[id] = u;	//The source supervertex
			if(id == New_EdgeSize-1){
				*gpu_new_VertexSize = u+1;	//Here u is the last supervertex
			}
		}
	}
}

//Update the  vertex list
__global__ void UpdateVertexList(unsigned int *gpu_vertex_list, unsigned int *gpu_vertex, unsigned int EdgeSize){
	unsigned int id=blockIdx.x*blockDim.x + threadIdx.x;
	if(id<EdgeSize){
		if(id>0){
			if(gpu_vertex_list[id]>gpu_vertex_list[id-1]){	//Marks the boundary of a new supervertex
				unsigned int index = gpu_vertex_list[id];	//The supervertex number is the vertex index in new vertex list
				gpu_vertex[index] = id;	//id is the number of elements so far, similar to inclusive scan output, which we need
			}
		}
		else{
			gpu_vertex[id]=0;	//Here id is 0. gpu_vertex[0] is always 0
		}
	}
}


//-------------------------------THE BORUVKA ALGORITHM IMPLEMENTATION----------------------------------------
void BoruvkaMST(){
	//NOTE: Used so many Kernels as a way to have global barriers between operations

	//Calculate number of blocks for running in parallel for length for |V| or |E|
	int Nblocks_edge=ceil(((float)EdgeSize)/1000);
	int Nblocks_vertices=ceil(((float)VertexSize)/1000);

	//Initialize the intermediate Arrays to 0
	InitiaizeZero<<<Nblocks_edge,1000>>>(gpu_vertex_list, EdgeSize);
	InitiaizeZero<<<Nblocks_edge,1000>>>(gpu_super_vertex_list, EdgeSize);

	//Output gpu_edge_WV has MSB 22 bits as edge dst vertex id and LSB 10 bits as weight
	ConcatenateWV<<<Nblocks_edge,1000>>>(gpu_edge_WV, gpu_edge_weight, gpu_edge_dst, EdgeSize);
	
	//Output gpu_min_edge has min edge for every edge segment for each vertex
	FindMin<<<Nblocks_vertices,1000>>>(gpu_min_edge,gpu_edge_WV,gpu_vertex,gpu_min_edge_index,gpu_edge_mapping,gpu_successor,VertexSize,EdgeSize);

	//Remove all cycles formed between pair of vertices
	RemoveCycles<<<Nblocks_vertices,1000>>>(gpu_successor,VertexSize);
		
	//Mark 1 at boundaries of edge segment for a vertex. Output flags in gpu_vertex_list
	InitiaizeZero<<<Nblocks_edge,1000>>>(gpu_vertex_list, EdgeSize);

	MarkSegment<<<Nblocks_vertices,1000>>>(gpu_vertex_list, gpu_vertex, VertexSize);
	
	//Do Inclusive Scan of the flags to get |E| size src vertex list for each edge. Output in gpu_vertex_list
	InclusiveScan<<<1,1>>>(gpu_vertex_list,EdgeSize);	//Use thrust on GPU so that no need to transfer data

	//Mark the min edge indexes found for the Final MST Edges
	OutputMST<<<Nblocks_vertices,1000>>>(gpu_OutputMST, gpu_successor, gpu_min_edge_index, VertexSize);

	//Copy succ to succ_temp. succ_temp is read only, to avoid any race condition
	CopyArray<<<Nblocks_vertices,1000>>>(gpu_successor_temp, gpu_successor, VertexSize);

	//Propogate the Representative Vertex IDs
	PropogateRepresentatives<<<Nblocks_vertices,1000>>>(gpu_successor, gpu_successor_temp, VertexSize);

	//Concatenate vertexes as (succ[i],i) form, will be used for sorting.
	ConcatenateSucc<<<Nblocks_vertices,1000>>>(gpu_sorted_vertex, gpu_successor, VertexSize);

	//Now sort the vertices according to (succ[i],i). In this way all vertices with same succ will be grouped together
	SortVertices<<<1,1>>>(gpu_sorted_vertex,VertexSize);

	//Mark Flags for Assigning Supervertices
	MarkSuperVertexFlags<<<Nblocks_vertices,1000>>>(gpu_super_vertex_list, gpu_sorted_vertex, VertexSize);

	//Now gpu_super_vertex_list will be a size |V| array containing the supervertex each vertex belongs to
	InclusiveScan<<<1,1>>>(gpu_super_vertex_list,VertexSize);

	//For each vertex assign the supervertex it belongs to
	AssignParentSuperVertex<<<Nblocks_vertices,1000>>>(gpu_parent_supervertex, gpu_super_vertex_list, gpu_sorted_vertex, VertexSize);

	//Get Number of supervertices. Later if no edge between any of them, they give count of trees in forest
	unsigned int *cpu_supervertex_temp;
	cpu_supervertex_temp = (unsigned int*)malloc(sizeof(unsigned int)*VertexSize);
	cudaMemcpy( cpu_supervertex_temp, gpu_super_vertex_list, sizeof(unsigned int)*VertexSize, cudaMemcpyDeviceToHost);
	unsigned int SuperVertex_count = cpu_supervertex_temp[VertexSize-1]+1;

	//Remove internal edges for vertices belonging to same supervertex
	RemoveInternalEdges<<<Nblocks_edge,1000>>>(gpu_edge_dst, gpu_vertex_list, gpu_parent_supervertex, EdgeSize);

	//Contains the updated list with concatenated (supervertex[i],i) supervertex for each edge at index i in edgelist between different supervertices
	MakeNewConcatnatedVertices<<<Nblocks_edge,1000>>>(gpu_new_concat_vertices, gpu_edge_dst, gpu_vertex_list, gpu_parent_supervertex, EdgeSize);
	
	//Now sort the concatenated list. Nullified edges will go to the end since they have highest value in their 32 MSB bits
	SortVertices<<<1,1>>>(gpu_new_concat_vertices,EdgeSize);

	//Updates the edge size
	UpdateEdgeSize<<<Nblocks_edge,1000>>>(gpu_new_concat_vertices, gpu_new_EdgeSize, EdgeSize);

	unsigned int New_EdgeSize;	//Copy the new edgelist size
	cudaMemcpy( &New_EdgeSize, gpu_new_EdgeSize, sizeof(unsigned int), cudaMemcpyDeviceToHost);
	if(New_EdgeSize==0){	//If no new edge remains between vertices
		flag=1;
		VertexSize=SuperVertex_count;
		return;
	}

	int Nblocks_edge_new=ceil(((float)New_EdgeSize)/1000);

	//Make a copy the necessary arrays, and use them as a read only version for MakeEdgeList kernel to prevent race condition
	CopyArray<<<Nblocks_edge,1000>>>(gpu_edge_mapping_temp, gpu_edge_mapping, EdgeSize);
	CopyArray<<<Nblocks_edge,1000>>>(gpu_edge_weight_temp, gpu_edge_weight, EdgeSize);
	CopyArray<<<Nblocks_edge,1000>>>(gpu_edge_dst_temp, gpu_edge_dst, EdgeSize);

	//Update the Edge List
	UpdateEdgeList<<<Nblocks_edge_new,1000>>>(gpu_new_concat_vertices, gpu_edge_mapping, gpu_edge_mapping_temp, gpu_edge_dst, gpu_edge_dst_temp,  gpu_parent_supervertex, gpu_edge_weight, gpu_edge_weight_temp, gpu_vertex_list, gpu_new_VertexSize, New_EdgeSize);

	unsigned int New_VertexSize;	//Copy the new Vertex List size
	cudaMemcpy( &New_VertexSize, gpu_new_VertexSize, sizeof(unsigned int), cudaMemcpyDeviceToHost);

	//Update the Vertexes in GPU_vertex
	UpdateVertexList<<<Nblocks_edge_new,1000>>>(gpu_vertex_list, gpu_vertex, New_EdgeSize);
	cudaDeviceSynchronize();

	//Set the new sizes for next iteration
	EdgeSize = New_EdgeSize;
	VertexSize = New_VertexSize;
}

//Serial Prims MST Algo. To compare results with our Parallel Boruvka implementation
long SerialMST_Prims(int r){
	int *visited = (int *)malloc(VertexSize_cpu*sizeof(int));	//Tells if that node is completed
	int *key = (int *)malloc(VertexSize_cpu*sizeof(int));	//Stores the key values of all vertices
	for(int i=0;i<VertexSize_cpu;i++){
		visited[i]=0;
		key[i]=INT_MAX;
	}
	key[r]=0;
	int Q_size=VertexSize_cpu;	//Tells number of vertices left to add in MST
	long MST_sum=0;
	int u=r;
	while(Q_size!=0){
        //Find the minimum
        int min_weight=INT_MAX;
        for(int i=0;i<VertexSize_cpu;i++){
            if(key[i]<min_weight && visited[i]==0){
                min_weight=key[i];    
                u=i;    //Note that always the lowest numbered vertice chosen for same weights
            }
		}
		visited[u]=1;
		Q_size--;
		MST_sum+=key[u];
		//Search its neighbours for the min among them
		int start=cpu_vertex[u], end=0;
		if(u==VertexSize_cpu-1) end = EdgeSize_cpu-1;
		else end = cpu_vertex[u+1]-1;
		for(int i=start;i<=end;i++){
			int v = cpu_edge_dst[i];
			if(visited[v]==0 && cpu_edge_weight[i]<key[v]){
				key[v]=cpu_edge_weight[i];
			}
		}
	}
	return MST_sum;	
}

//Used to generate a large dataset
void Generate_Random_Graph(char *filename, int vertices_count){
	int v = vertices_count;
	int e = rand()%((v*(v-1)/2)-1)+1;
    FILE *fptr;
    fptr = fopen(filename , "w");
	set< pair<int,int> > S;
	fprintf(fptr, "%d " , v);
	fprintf(fptr, "%d\n" , e);
	int iter=0;
	int a = rand()%v;
	int b = rand()%v;
	int wt = rand()%1000;
	S.insert(make_pair(a,b));
	S.insert(make_pair(b,a));
	fprintf(fptr, "%d " , a);
	fprintf(fptr, "%d " , b);
	fprintf(fptr, "%d\n" , wt);
	iter++;
    //To Generate a forest
	while(iter<e){
		a = rand()%v;
		b = rand()%v;
		wt = rand()%1000;
		//Check if that edge already declared
		if((a!=b) && (S.find(make_pair(a,b))==S.end() || S.find(make_pair(b,a))==S.end())){
			S.insert(make_pair(a,b));
			S.insert(make_pair(b,a));
			fprintf(fptr, "%d " , a);
			fprintf(fptr, "%d " , b);
			fprintf(fptr, "%d\n" , wt);
			iter++;
		}
	}
	fclose(fptr);
}

int main(int argc,char **argv){
	printf("\n------------------------------------\n");
    //--------------------------------READ THE INPUT GRAPH-------------------------------------
	
	//Note that input must be of form: 1st line: n vertices, m edges. Next m lines must have m edges in form of: src dst wt
    FILE *inputfilepointer;
    char *inputfilename = argv[1];
    inputfilepointer    = fopen(inputfilename , "r");
    if ( inputfilepointer == NULL )  {
        printf( "Input file failed to open." );
        return 0;
    }
	printf("Input File Given: %s\n",argv[1]);
	
	//Generate_Random_Graph(argv[1],1000);	//Generates a randomly large dataset
	
	fscanf(inputfilepointer,"%d",&VertexSize); 
	VertexSize_cpu = VertexSize ;
	fscanf(inputfilepointer,"%d",&EdgeSize);
	EdgeSize=EdgeSize*2; 
	EdgeSize_cpu = EdgeSize ;
	vector < pair<pair<unsigned int,unsigned int>,unsigned int> > EdgeList;
	unsigned int u,v,wt;	//u->src, v->dst of an edge
	unsigned int *cpu_outdegree = (unsigned int*)malloc(sizeof(unsigned int)*VertexSize);
	for(int i=0;i<VertexSize; i++){
		cpu_outdegree[i]=0;
	}
	for(int i=0; i<EdgeSize/2; i++){
		fscanf(inputfilepointer,"%d",&u); 
		fscanf(inputfilepointer,"%d",&v); 
		fscanf(inputfilepointer,"%d",&wt); 
		//u--;	v--;	//In asgn2 input files, vertex index start from 1. #Change this according to input
		EdgeList.push_back(make_pair(make_pair(u,v),wt));
		EdgeList.push_back(make_pair(make_pair(v,u),wt));
		cpu_outdegree[u]++;
		cpu_outdegree[v]++;
	}
	sort(EdgeList.begin(),EdgeList.end());
	cpu_vertex = (unsigned int*)malloc(sizeof(unsigned int)*VertexSize);
	for(int i=0;i<VertexSize; i++){
		cpu_vertex[i]=cpu_outdegree[i];
	}
	thrust::exclusive_scan(thrust::host,cpu_vertex,cpu_vertex+VertexSize,cpu_vertex);
	cpu_edge_dst = (unsigned int*) malloc (sizeof(unsigned int)*EdgeSize);
	cpu_edge_weight = (unsigned int*) malloc (sizeof(unsigned int)*EdgeSize);
	for(int i=0;i<EdgeSize; i++){
		cpu_edge_dst[i] = ((EdgeList.at(i)).first).second;
		cpu_edge_weight[i] = (EdgeList.at(i)).second;
	}
    fclose(inputfilepointer);

	//Check if input Graph is connected
	for(int i=1; i<VertexSize; i++){
		if(cpu_vertex[i]==cpu_vertex[i-1]){
			printf("INPUT GRAPH IS NOT COMPLETE. GIVE A COMPLETE CONNECTED GRAPH AS INPUT.\nCheck Node %d\n----EXITING----\n",i);
			return 0;
		}
	}
	
    //---------------------------INITIALIZE GPU MEMORY----------------------------------------
    
    unsigned int GPU_Mem=0; //Stores total memory allocated on GPU
    
    //Copy the Graph to Device
    cudaMalloc(&gpu_vertex, sizeof(unsigned int)*VertexSize);
	cudaMemcpy(gpu_vertex, cpu_vertex, sizeof(unsigned int)*VertexSize, cudaMemcpyHostToDevice);
	GPU_Mem+=sizeof(unsigned int)*VertexSize;
	
	cudaMalloc(&gpu_edge_dst, sizeof(unsigned int)*EdgeSize);
	cudaMemcpy(gpu_edge_dst, cpu_edge_dst, sizeof(unsigned int)*EdgeSize, cudaMemcpyHostToDevice);
	GPU_Mem+=sizeof(unsigned int)*EdgeSize;
	
    cudaMalloc(&gpu_edge_weight, sizeof(unsigned int)*EdgeSize);
	cudaMemcpy(gpu_edge_weight, cpu_edge_weight, sizeof(unsigned int)*EdgeSize, cudaMemcpyHostToDevice);
	GPU_Mem+=sizeof(unsigned int)*EdgeSize;

	cudaMalloc(&gpu_edge_weight_temp, sizeof(unsigned int)*EdgeSize);
	GPU_Mem+=sizeof(unsigned int)*EdgeSize;

	cudaMalloc(&gpu_edge_dst_temp, sizeof(unsigned int)*EdgeSize);
	GPU_Mem+=sizeof(unsigned int)*EdgeSize;

	cudaMalloc(&gpu_edge_WV, sizeof(unsigned int)*EdgeSize);
	cudaMemset(&gpu_edge_WV,0,sizeof(unsigned int)*EdgeSize);
	GPU_Mem+=sizeof(unsigned int)*EdgeSize;

    cudaMalloc(&gpu_edge_mapping, sizeof(unsigned int)*EdgeSize);
	unsigned int *temp_edgesize = (unsigned int*)malloc(sizeof(unsigned int)*EdgeSize);
	for(int i=0;i<EdgeSize;i++)temp_edgesize[i]=i;
	cudaMemcpy(gpu_edge_mapping, temp_edgesize, sizeof(unsigned int)*EdgeSize, cudaMemcpyHostToDevice);
	GPU_Mem+=sizeof(unsigned int)*EdgeSize;

    cudaMalloc(&gpu_edge_mapping_temp, sizeof(unsigned int)*EdgeSize);
	GPU_Mem+=sizeof(unsigned int)*EdgeSize;

    cudaMalloc(&gpu_min_edge, sizeof(unsigned int)*VertexSize);
	GPU_Mem+=sizeof(unsigned int)*VertexSize;

    cudaMalloc(&gpu_min_edge_index, sizeof(unsigned int)*VertexSize);
	GPU_Mem+=sizeof(unsigned int)*VertexSize;

    cudaMalloc(&gpu_successor, sizeof(unsigned int)*VertexSize);
	GPU_Mem+=sizeof(unsigned int)*VertexSize;

    cudaMalloc(&gpu_successor_temp, sizeof(unsigned int)*VertexSize);
	GPU_Mem+=sizeof(unsigned int)*VertexSize;

    cudaMalloc(&gpu_vertex_list, sizeof(unsigned int)*EdgeSize);
	GPU_Mem+=sizeof(unsigned int)*EdgeSize;

	cudaMalloc(&gpu_OutputMST, sizeof(unsigned int)*EdgeSize);
	//cudaMemset(&gpu_OutputMST,0,sizeof(unsigned int)*EdgeSize);
	int Nblocks_edge=ceil(((float)EdgeSize)/1000);
	InitiaizeZero<<<Nblocks_edge,1000>>>(gpu_OutputMST,EdgeSize);
	GPU_Mem+=sizeof(unsigned int)*EdgeSize;

    cudaMalloc(&gpu_sorted_vertex, sizeof(unsigned long)*VertexSize);
	GPU_Mem+=sizeof(unsigned long)*VertexSize;

    cudaMalloc(&gpu_super_vertex_list, sizeof(unsigned int)*VertexSize);
	GPU_Mem+=sizeof(unsigned int)*VertexSize;

    cudaMalloc(&gpu_parent_supervertex, sizeof(unsigned int)*VertexSize);
	GPU_Mem+=sizeof(unsigned int)*VertexSize;

	cudaMalloc(&gpu_new_concat_vertices, sizeof(unsigned long)*EdgeSize);
	GPU_Mem+=sizeof(unsigned int)*EdgeSize;

	cudaMalloc(&gpu_new_EdgeSize, sizeof(unsigned int));
	GPU_Mem+=sizeof(unsigned int);

	cudaMalloc(&gpu_new_VertexSize, sizeof(unsigned int));
	GPU_Mem+=sizeof(unsigned int);

	printf("Total Memory Occupied on GPU : %.3f MB\n",(float)GPU_Mem/(1024*1024));
    //---------------START MST COMPUTATION USING OUR BORUVKA ALGORTIHM------------------

	clock_t start_time = clock();
	flag=0;
    //Recursively call until only one vertex remains
	while(VertexSize>1 && flag==0){
		BoruvkaMST();
	}
	
	clock_t end_time = clock();
    printf("Time taken for the Boruvka MST Algorithm: %0.3f ms\n",((double)(end_time-start_time)/1000));

	//-------------------PRINT THE OUTPUT------------------------------------------------
	//cpu_vertex_list is of size |E| and contains src vertex id for every edge
	cpu_vertex_list = (unsigned int *)malloc(sizeof(unsigned int)*EdgeSize_cpu);
	unsigned int *cpu_vertex_list_flag = (unsigned int *)malloc(sizeof(unsigned int)*EdgeSize_cpu);
	for(int i=1;i<EdgeSize_cpu;i++)
		cpu_vertex_list_flag[i]=0;
	for(int i=0;i<VertexSize_cpu;i++){
		cpu_vertex_list_flag[cpu_vertex[i]]=1;
	}
	cpu_vertex_list[0]=0;
	for(int i=1;i<EdgeSize_cpu;i++){
		if(cpu_vertex_list_flag[i]==1)
			cpu_vertex_list[i] = cpu_vertex_list[i-1]+1;
		else
			cpu_vertex_list[i] = cpu_vertex_list[i-1];
	}
	unsigned int *cpu_OutputMST = (unsigned int*)malloc(sizeof(unsigned int)*EdgeSize_cpu);
	cudaMemcpy( cpu_OutputMST, gpu_OutputMST, sizeof(unsigned int)*EdgeSize_cpu, cudaMemcpyDeviceToHost);
	unsigned int MST_EdgeSize=0;
	unsigned long MST_weight=0;
	printf("\n  BORUVKA MST RESULTS:\n");
	//printf(" src\tdst\tweight\n");
	for(int i=0; i<EdgeSize_cpu; i++){
		if(cpu_OutputMST[i]==1){
			MST_EdgeSize++;
			MST_weight+=cpu_edge_weight[i];
			//printf(" %d\t%d\t%d\n",cpu_vertex_list[i],cpu_edge_dst[i],cpu_edge_weight[i]);
		}
	}
	printf("Number of Edges: %d \t(Expected: %d)\n",MST_EdgeSize,VertexSize_cpu-1);
	printf("MST sum of Weights: %lu\n\n",MST_weight);

	//---------------------FREE THE GPU MEMORY--------------------------------------------------------------
	cudaFree(gpu_vertex);
	cudaFree(gpu_edge_dst);
	cudaFree(gpu_edge_weight);
	cudaFree(gpu_edge_weight_temp);
	cudaFree(gpu_edge_WV);
	cudaFree(gpu_edge_mapping);
	cudaFree(gpu_edge_mapping_temp);
	cudaFree(gpu_min_edge);
	cudaFree(gpu_min_edge_index);
	cudaFree(gpu_successor);
	cudaFree(gpu_successor_temp);
	cudaFree(gpu_vertex_list);
	cudaFree(gpu_OutputMST);
	cudaFree(gpu_sorted_vertex);
	cudaFree(gpu_super_vertex_list);
	cudaFree(gpu_parent_supervertex);
	cudaFree(gpu_new_concat_vertices);
	cudaFree(gpu_new_EdgeSize);
	cudaFree(gpu_new_VertexSize);

	//------------------COMPARE WITH SERIAL MST ALGORITHMS--------------------------------------------

	start_time = clock();
	long MST_sum_Prims = SerialMST_Prims(0);	//Call Prims with vertex 0 as starting vertex
	end_time = clock();
	printf("\nTime taken by Serial Prims Algorithm = %0.3f ms\n",((double)(end_time-start_time)/1000));
	printf("MST sum of Weights by Prims: %lu\n",MST_sum_Prims);

	if(MST_weight==MST_sum_Prims){
		printf("RESULTS MATCH!! SUCCESS\n");
		printf("---------------------------\n\n");
	}
    return 0;
}
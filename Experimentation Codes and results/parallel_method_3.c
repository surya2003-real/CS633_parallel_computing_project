/*
 * MPI-Based 3D Data Reader and Halo Exchange
 * --------------------------------------------
 * This program reads a 3D volumetric dataset distributed among MPI processes,
 * performs halo exchange, and computes local/global minima and maxima.
 * Output: local/global statistics and timing data.
 *
 * All debug print statements have been removed.
 */

 #include <stdio.h>
 #include <stdlib.h>
 #include <mpi.h>
 #include <limits.h>
 #include <float.h>
 
 int PX;
 int PY;
 int PZ;
 int NX;
 int NY;
 int NZ;
 int NC;
 
 /*
  * exchange_halo:
  *   Performs a halo exchange for a 4D array (indexed as [x][y][z][channel]) among adjacent
  *   MPI processes arranged in a 3D Cartesian grid. The halo regions are exchanged along all six faces.
  */
 void exchange_halo(float ****halo, int local_nx, int local_ny, int local_nz, int NC,
                    int rank, int PX, int PY, int PZ, MPI_Comm comm)
 {
     // Compute process grid coordinates
     int px = rank % PX;
     int py = (rank / PX) % PY;
     int pz = rank / (PX * PY);
 
     // Determine neighboring ranks for each face
     int left   = (px > 0)            ? rank - 1          : MPI_PROC_NULL;
     int right  = (px < PX - 1)       ? rank + 1          : MPI_PROC_NULL;
     int top    = (py > 0)            ? rank - PX         : MPI_PROC_NULL;
     int bottom = (py < PY - 1)       ? rank + PX         : MPI_PROC_NULL;
     int front  = (pz > 0)            ? rank - PX * PY    : MPI_PROC_NULL;
     int back   = (pz < PZ - 1)       ? rank + PX * PY    : MPI_PROC_NULL;
 
     MPI_Request reqs[12];
     int req_count = 0;
 
     /* Allocate buffers for the 6 faces (send and receive for each) */
     float *send_left   = (float *)malloc(local_ny * local_nz * NC * sizeof(float));
     float *recv_left   = (float *)malloc(local_ny * local_nz * NC * sizeof(float));
     float *send_right  = (float *)malloc(local_ny * local_nz * NC * sizeof(float));
     float *recv_right  = (float *)malloc(local_ny * local_nz * NC * sizeof(float));
     float *send_front  = (float *)malloc(local_nx * local_ny * NC * sizeof(float));
     float *recv_front  = (float *)malloc(local_nx * local_ny * NC * sizeof(float));
     float *send_back   = (float *)malloc(local_nx * local_ny * NC * sizeof(float));
     float *recv_back   = (float *)malloc(local_nx * local_ny * NC * sizeof(float));
     float *send_bottom = (float *)malloc(local_nx * local_nz * NC * sizeof(float));
     float *recv_bottom = (float *)malloc(local_nx * local_nz * NC * sizeof(float));
     float *send_top    = (float *)malloc(local_nx * local_nz * NC * sizeof(float));
     float *recv_top    = (float *)malloc(local_nx * local_nz * NC * sizeof(float));
 
     /* Exchange Left-Right halos */
     if (left != MPI_PROC_NULL) {
         for (int y = 1; y <= local_ny; y++) {
             for (int z = 1; z <= local_nz; z++) {
                 for (int t = 0; t < NC; t++) {
                     // Pack left face from halo region at index 1
                     send_left[(y - 1) * local_nz * NC + (z - 1) * NC + t] = halo[1][y][z][t];
                 }
             }
         }
         MPI_Isend(send_left, local_ny * local_nz * NC, MPI_FLOAT, left, 0, comm, &reqs[req_count++]);
         MPI_Irecv(recv_left, local_ny * local_nz * NC, MPI_FLOAT, left, 1, comm, &reqs[req_count++]);
         MPI_Wait(&reqs[req_count - 1], MPI_STATUS_IGNORE);
         MPI_Wait(&reqs[req_count - 2], MPI_STATUS_IGNORE);
         for (int y = 1; y <= local_ny; y++)
             for (int z = 1; z <= local_nz; z++)
                 for (int t = 0; t < NC; t++)
                     halo[0][y][z][t] = recv_left[(y - 1) * local_nz * NC + (z - 1) * NC + t];
     }
     req_count = 0;
     if (right != MPI_PROC_NULL) {
         for (int y = 1; y <= local_ny; y++)
             for (int z = 1; z <= local_nz; z++)
                 for (int t = 0; t < NC; t++)
                     send_right[(y - 1) * local_nz * NC + (z - 1) * NC + t] = halo[local_nx][y][z][t];
         MPI_Isend(send_right, local_ny * local_nz * NC, MPI_FLOAT, right, 1, comm, &reqs[req_count++]);
         MPI_Irecv(recv_right, local_ny * local_nz * NC, MPI_FLOAT, right, 0, comm, &reqs[req_count++]);
         MPI_Wait(&reqs[req_count - 1], MPI_STATUS_IGNORE);
         MPI_Wait(&reqs[req_count - 2], MPI_STATUS_IGNORE);
         for (int y = 1; y <= local_ny; y++)
             for (int z = 1; z <= local_nz; z++)
                 for (int t = 0; t < NC; t++)
                     halo[local_nx + 1][y][z][t] = recv_right[(y - 1) * local_nz * NC + (z - 1) * NC + t];
     }
     req_count = 0;
     /* Exchange Front-Back halos */
     if (front != MPI_PROC_NULL) {
         for (int x = 1; x <= local_nx; x++)
             for (int y = 1; y <= local_ny; y++)
                 for (int t = 0; t < NC; t++)
                     send_front[(x - 1) * local_ny * NC + (y - 1) * NC + t] = halo[x][y][1][t];
         MPI_Isend(send_front, local_nx * local_ny * NC, MPI_FLOAT, front, 2, comm, &reqs[req_count++]);
         MPI_Irecv(recv_front, local_nx * local_ny * NC, MPI_FLOAT, front, 3, comm, &reqs[req_count++]);
         MPI_Wait(&reqs[req_count - 1], MPI_STATUS_IGNORE);
         MPI_Wait(&reqs[req_count - 2], MPI_STATUS_IGNORE);
         for (int x = 1; x <= local_nx; x++)
             for (int y = 1; y <= local_ny; y++)
                 for (int t = 0; t < NC; t++)
                     halo[x][y][0][t] = recv_front[(x - 1) * local_ny * NC + (y - 1) * NC + t];
     }
     req_count = 0;
     if (back != MPI_PROC_NULL) {
         for (int x = 1; x <= local_nx; x++)
             for (int y = 1; y <= local_ny; y++)
                 for (int t = 0; t < NC; t++)
                     send_back[(x - 1) * local_ny * NC + (y - 1) * NC + t] = halo[x][y][local_nz][t];
         MPI_Isend(send_back, local_nx * local_ny * NC, MPI_FLOAT, back, 3, comm, &reqs[req_count++]);
         MPI_Irecv(recv_back, local_nx * local_ny * NC, MPI_FLOAT, back, 2, comm, &reqs[req_count++]);
         MPI_Wait(&reqs[req_count - 1], MPI_STATUS_IGNORE);
         MPI_Wait(&reqs[req_count - 2], MPI_STATUS_IGNORE);
         for (int x = 1; x <= local_nx; x++)
             for (int y = 1; y <= local_ny; y++)
                 for (int t = 0; t < NC; t++)
                     halo[x][y][local_nz + 1][t] = recv_back[(x - 1) * local_ny * NC + (y - 1) * NC + t];
     }
     req_count = 0;
     /* Exchange Top-Bottom halos */
     if (top != MPI_PROC_NULL) {
         for (int x = 1; x <= local_nx; x++)
             for (int z = 1; z <= local_nz; z++)
                 for (int t = 0; t < NC; t++)
                     send_top[(x - 1) * local_nz * NC + (z - 1) * NC + t] = halo[x][1][z][t];
         MPI_Isend(send_top, local_nx * local_nz * NC, MPI_FLOAT, top, 4, comm, &reqs[req_count++]);
         MPI_Irecv(recv_top, local_nx * local_nz * NC, MPI_FLOAT, top, 5, comm, &reqs[req_count++]);
         MPI_Wait(&reqs[req_count - 1], MPI_STATUS_IGNORE);
         MPI_Wait(&reqs[req_count - 2], MPI_STATUS_IGNORE);
         for (int x = 1; x <= local_nx; x++)
             for (int z = 1; z <= local_nz; z++)
                 for (int t = 0; t < NC; t++)
                     halo[x][0][z][t] = recv_top[(x - 1) * local_nz * NC + (z - 1) * NC + t];
     }
     req_count = 0;
     if (bottom != MPI_PROC_NULL) {
         for (int x = 1; x <= local_nx; x++)
             for (int z = 1; z <= local_nz; z++)
                 for (int t = 0; t < NC; t++)
                     send_bottom[(x - 1) * local_nz * NC + (z - 1) * NC + t] = halo[x][local_ny][z][t];
         MPI_Isend(send_bottom, local_nx * local_nz * NC, MPI_FLOAT, bottom, 5, comm, &reqs[req_count++]);
         MPI_Irecv(recv_bottom, local_nx * local_nz * NC, MPI_FLOAT, bottom, 4, comm, &reqs[req_count++]);
         MPI_Wait(&reqs[req_count - 1], MPI_STATUS_IGNORE);
         MPI_Wait(&reqs[req_count - 2], MPI_STATUS_IGNORE);
         for (int x = 1; x <= local_nx; x++)
             for (int z = 1; z <= local_nz; z++)
                 for (int t = 0; t < NC; t++)
                     halo[x][local_ny + 1][z][t] = recv_bottom[(x - 1) * local_nz * NC + (z - 1) * NC + t];
     }
 
     /* Free halo exchange buffers */
     free(send_left);
     free(recv_left);
     free(send_right);
     free(recv_right);
     free(send_front);
     free(recv_front);
     free(send_back);
     free(recv_back);
     free(send_bottom);
     free(recv_bottom);
     free(send_top);
     free(recv_top);
 }
 
 /*
  * find_local_minima:
  *   Iterates over each local grid point (excluding the halo) and finds the minima for each channel.
  *   A grid point is a local minimum if its value is lower than all of its adjacent neighbors, including halo regions.
  */
 void find_local_minima(int rank, int PX, int PY, int PZ, float ****halo,
                        int local_nx, int local_ny, int local_nz,
                        float **local_minima, int *min_count)
 {
     int px = rank % PX;
     int py = (rank / PX) % PY;
     int pz = rank / (PX * PY);
 
     int left   = (px > 0)            ? rank - 1          : MPI_PROC_NULL;
     int right  = (px < PX - 1)       ? rank + 1          : MPI_PROC_NULL;
     int top    = (py > 0)            ? rank - PX         : MPI_PROC_NULL;
     int bottom = (py < PY - 1)       ? rank + PX         : MPI_PROC_NULL;
     int front  = (pz > 0)            ? rank - PX * PY    : MPI_PROC_NULL;
     int back   = (pz < PZ - 1)       ? rank + PX * PY    : MPI_PROC_NULL;
 
     for (int i = 1; i <= local_nx; i++) {
         for (int j = 1; j <= local_ny; j++) {
             for (int k = 1; k <= local_nz; k++) {
                 for (int t = 0; t < NC; t++) {
                     float val = halo[i][j][k][t];
                     int is_min = 1;
                     if (i > 1 && val >= halo[i - 1][j][k][t])
                         is_min = 0;
                     if (i < local_nx && val >= halo[i + 1][j][k][t])
                         is_min = 0;
                     if (j > 1 && val >= halo[i][j - 1][k][t])
                         is_min = 0;
                     if (j < local_ny && val >= halo[i][j + 1][k][t])
                         is_min = 0;
                     if (k > 1 && val >= halo[i][j][k - 1][t])
                         is_min = 0;
                     if (k < local_nz && val >= halo[i][j][k + 1][t])
                         is_min = 0;
                     if (i == 1 && left != MPI_PROC_NULL && val >= halo[i - 1][j][k][t])
                         is_min = 0;
                     if (i == local_nx && right != MPI_PROC_NULL && val >= halo[i + 1][j][k][t])
                         is_min = 0;
                     if (j == 1 && top != MPI_PROC_NULL && val >= halo[i][j - 1][k][t])
                         is_min = 0;
                     if (j == local_ny && bottom != MPI_PROC_NULL && val >= halo[i][j + 1][k][t])
                         is_min = 0;
                     if (k == 1 && front != MPI_PROC_NULL && val >= halo[i][j][k - 1][t])
                         is_min = 0;
                     if (k == local_nz && back != MPI_PROC_NULL && val >= halo[i][j][k + 1][t])
                         is_min = 0;
                     if (is_min) {
                         local_minima[t][min_count[t]] = val;
                         (min_count[t])++;
                     }
                 }
             }
         }
     }
 }
 
 /*
  * find_local_maxima:
  *   Iterates over each local grid point (excluding the halo) and finds the maxima for each channel.
  *   A grid point is a local maximum if its value is higher than all of its adjacent neighbors, including halo regions.
  */
 void find_local_maxima(int rank, int PX, int PY, int PZ, float ****halo,
                        int local_nx, int local_ny, int local_nz,
                        float **local_maxima, int *max_count)
 {
     int px = rank % PX;
     int py = (rank / PX) % PY;
     int pz = rank / (PX * PY);
 
     int left   = (px > 0)            ? rank - 1          : MPI_PROC_NULL;
     int right  = (px < PX - 1)       ? rank + 1          : MPI_PROC_NULL;
     int top    = (py > 0)            ? rank - PX         : MPI_PROC_NULL;
     int bottom = (py < PY - 1)       ? rank + PX         : MPI_PROC_NULL;
     int front  = (pz > 0)            ? rank - PX * PY    : MPI_PROC_NULL;
     int back   = (pz < PZ - 1)       ? rank + PX * PY    : MPI_PROC_NULL;
 
     for (int i = 1; i <= local_nx; i++) {
         for (int j = 1; j <= local_ny; j++) {
             for (int k = 1; k <= local_nz; k++) {
                 for (int t = 0; t < NC; t++) {
                     float val = halo[i][j][k][t];
                     int is_max = 1;
                     if (i > 1 && val <= halo[i - 1][j][k][t])
                         is_max = 0;
                     if (i < local_nx && val <= halo[i + 1][j][k][t])
                         is_max = 0;
                     if (j > 1 && val <= halo[i][j - 1][k][t])
                         is_max = 0;
                     if (j < local_ny && val <= halo[i][j + 1][k][t])
                         is_max = 0;
                     if (k > 1 && val <= halo[i][j][k - 1][t])
                         is_max = 0;
                     if (k < local_nz && val <= halo[i][j][k + 1][t])
                         is_max = 0;
                     if (i == 1 && left != MPI_PROC_NULL && val <= halo[i - 1][j][k][t])
                         is_max = 0;
                     if (i == local_nx && right != MPI_PROC_NULL && val <= halo[i + 1][j][k][t])
                         is_max = 0;
                     if (j == 1 && top != MPI_PROC_NULL && val <= halo[i][j - 1][k][t])
                         is_max = 0;
                     if (j == local_ny && bottom != MPI_PROC_NULL && val <= halo[i][j + 1][k][t])
                         is_max = 0;
                     if (k == 1 && front != MPI_PROC_NULL && val <= halo[i][j][k - 1][t])
                         is_max = 0;
                     if (k == local_nz && back != MPI_PROC_NULL && val <= halo[i][j][k + 1][t])
                         is_max = 0;
                     if (is_max) {
                         local_maxima[t][max_count[t]] = val;
                         (max_count[t])++;
                     }
                 }
             }
         }
     }
 }
 
 /*
  * read_and_scatter_data:
  *   Uses MPI-IO to read a subarray from the global data file into a local buffer for each MPI process.
  *   The file view is set to extract only the process's block based on its position in the process grid.
  */
 void read_and_scatter_data(float *local_data, const char *input_file,
                            int local_nx, int local_ny, int local_nz)
 {
     MPI_File fh;
     MPI_Status status;
     MPI_Datatype filetype;
 
     // Global array dimensions (order: Z, Y, X, channels)
     int global_sizes[4] = {NZ, NY, NX, NC};
 
     // Local subarray dimensions (each process's block)
     int local_sizes[4] = {local_nz, local_ny, local_nx, NC};
 
     int rank;
     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
 
     // Compute grid coordinates from rank
     int rank_x = rank % PX;
     int rank_y = (rank / PX) % PY;
     int rank_z = rank / (PX * PY);
 
     // Starting indices in the global data for this process's block.
     int starts[4] = {rank_z * local_nz,
                      rank_y * local_ny,
                      rank_x * local_nx,
                      0};
 
     /* Create subarray datatype corresponding to this block */
     MPI_Type_create_subarray(4,
                              global_sizes,
                              local_sizes,
                              starts,
                              MPI_ORDER_C,
                              MPI_FLOAT,
                              &filetype);
     MPI_Type_commit(&filetype);
 
     /* Open the input file collectively */
     int rc = MPI_File_open(MPI_COMM_WORLD, input_file,
                            MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
     if (rc != MPI_SUCCESS)
     {
         fprintf(stderr, "Error opening file %s\n", input_file);
         MPI_Abort(MPI_COMM_WORLD, rc);
     }
 
     /* Set file view to restrict each process to its own subarray */
     MPI_File_set_view(fh,
                       0,
                       MPI_FLOAT,
                       filetype,
                       "native",
                       MPI_INFO_NULL);
 
     int local_count = local_nx * local_ny * local_nz * NC;
 
     /* Read the local subarray into local_data */
     MPI_File_read_all(fh, local_data, local_count, MPI_FLOAT, &status);
 
     MPI_File_close(&fh);
     MPI_Type_free(&filetype);
 }
 
 int main(int argc, char **argv)
 {
     MPI_Init(&argc, &argv);
 
     int rank, num_procs;
     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
     MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
 
     if (argc != 10)
     {
         if (rank == 0)
         {
             fprintf(stderr, "Usage: %s <input_file> PX PY PZ NX NY NZ NC <output_file>\n", argv[0]);
         }
         MPI_Finalize();
         return EXIT_FAILURE;
     }
 
     char *input_file = argv[1];
     PX = atoi(argv[2]);
     PY = atoi(argv[3]);
     PZ = atoi(argv[4]);
     NX = atoi(argv[5]);
     NY = atoi(argv[6]);
     NZ = atoi(argv[7]);
     NC = atoi(argv[8]);
     char *output_file = argv[9];
 
     int local_nx = NX / PX, local_ny = NY / PY, local_nz = NZ / PZ;
     int px = rank % PX;
     int py = (rank / PX) % PY;
     int pz = rank / (PX * PY);
 
     // Adjust local dimensions for processes on the grid boundaries
     int right = (px < PX - 1) ? rank + 1 : MPI_PROC_NULL;
     int bottom = (py < PY - 1) ? rank + PX : MPI_PROC_NULL;
     int back = (pz < PZ - 1) ? rank + PX * PY : MPI_PROC_NULL;
 
     if (right == MPI_PROC_NULL && (NX % PX))
         local_nx += NX % PX;
     if (bottom == MPI_PROC_NULL && (NY % PY))
         local_ny += NY % PY;
     if (back == MPI_PROC_NULL && (NZ % PZ))
         local_nz += NZ % PZ;
 
     double startime, endtime;
     startime = MPI_Wtime();
 
     float *local_data = (float *)malloc(local_nx * local_ny * local_nz * NC * sizeof(float));
     read_and_scatter_data(local_data, input_file, local_nx, local_ny, local_nz);
 
     double data_read_time = MPI_Wtime() - startime;
     double max_data_read_time;
     MPI_Reduce(&data_read_time, &max_data_read_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
 
     /* Allocate memory and unpack local_data into a 4D halo array with halo boundaries */
     float ****halo = (float ****)malloc((local_nx + 2) * sizeof(float ***));
     for (int i = 0; i < (local_nx + 2); i++)
     {
         halo[i] = (float ***)malloc((local_ny + 2) * sizeof(float **));
         for (int j = 0; j < (local_ny + 2); j++)
         {
             halo[i][j] = (float **)malloc((local_nz + 2) * sizeof(float *));
             for (int k = 0; k < (local_nz + 2); k++)
             {
                 halo[i][j][k] = (float *)malloc(NC * sizeof(float));
             }
         }
     }
 
     /* Unpack the received local data into the halo array (excluding halo boundaries) */
     int position = 0;
     for (int z = 1; z <= local_nz; z++)
     {
         for (int y = 1; y <= local_ny; y++)
         {
             for (int x = 1; x <= local_nx; x++)
             {
                 for (int t = 0; t < NC; t++)
                 {
                     MPI_Unpack(local_data, local_nx * local_ny * local_nz * NC, &position,
                                &halo[x][y][z][t], 1, MPI_FLOAT, MPI_COMM_WORLD);
                 }
             }
         }
     }
 
     /* Perform Halo Exchange */
     exchange_halo(halo, local_nx, local_ny, local_nz, NC, rank, PX, PY, PZ, MPI_COMM_WORLD);
 
     /* Compute local minima for each channel */
     float **local_minima = (float **)malloc(NC * sizeof(float *));
     for (int i = 0; i < NC; i++)
     {
         local_minima[i] = (float *)malloc(local_nx * local_ny * local_nz * sizeof(float));
     }
     int *min_count = (int *)malloc(NC * sizeof(int));
     for (int i = 0; i < NC; i++)
         min_count[i] = 0;
     find_local_minima(rank, PX, PY, PZ, halo, local_nx, local_ny, local_nz, local_minima, min_count);
 
     /* Compute local maxima for each channel */
     float **local_maxima = (float **)malloc(NC * sizeof(float *));
     for (int i = 0; i < NC; i++)
     {
         local_maxima[i] = (float *)malloc(local_nx * local_ny * local_nz * sizeof(float));
     }
     int *max_count = (int *)malloc(NC * sizeof(int));
     for (int i = 0; i < NC; i++)
         max_count[i] = 0;
     find_local_maxima(rank, PX, PY, PZ, halo, local_nx, local_ny, local_nz, local_maxima, max_count);
 
     /* Compute global minima and maxima via reduction */
     float *rank_minima = (float *)malloc(NC * sizeof(float));
     float *rank_maxima = (float *)malloc(NC * sizeof(float));
     for (int i = 0; i < NC; i++)
     {
         rank_minima[i] = FLT_MAX;
         rank_maxima[i] = FLT_MIN;
     }
     for (int i = 0; i < NC; i++)
     {
         for (int j = 0; j < min_count[i]; j++)
         {
             if (local_minima[i][j] < rank_minima[i])
                 rank_minima[i] = local_minima[i][j];
         }
         for (int j = 0; j < max_count[i]; j++)
         {
             if (local_maxima[i][j] > rank_maxima[i])
                 rank_maxima[i] = local_maxima[i][j];
         }
     }
     float *global_minima = (float *)malloc(NC * sizeof(float));
     MPI_Reduce(rank_minima, global_minima, NC, MPI_FLOAT, MPI_MIN, 0, MPI_COMM_WORLD);
     float *global_maxima = (float *)malloc(NC * sizeof(float));
     MPI_Reduce(rank_maxima, global_maxima, NC, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);
     int *total_local_minima = (int *)malloc(NC * sizeof(int));
     MPI_Reduce(min_count, total_local_minima, NC, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
     int *total_local_maxima = (int *)malloc(NC * sizeof(int));
     MPI_Reduce(max_count, total_local_maxima, NC, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
 
     /* Final timing and output writing (writing results to output file) */
     endtime = MPI_Wtime();
     double totaltime = endtime - startime;
     double global_totaltime;
     MPI_Reduce(&totaltime, &global_totaltime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
 
     if (rank == 0)
     {
         FILE *output = fopen(output_file, "w");
         if (!output)
         {
             perror("Error opening output file");
             MPI_Abort(MPI_COMM_WORLD, 1);
         }
         // Line 1: (local minima count, local maxima count) for each channel
         for (int i = 0; i < NC; i++)
         {
             fprintf(output, "(%d, %d), ", total_local_minima[i], total_local_maxima[i]);
         }
         fprintf(output, "\n");
         // Line 2: (global minimum, global maximum) for each channel
         for (int i = 0; i < NC; i++)
         {
             fprintf(output, "(%.4f, %.4f), ", global_minima[i], global_maxima[i]);
         }
         fprintf(output, "\n");
         // Line 3: Data read time, main code time, and total time
         fprintf(output, "%f, %f, %f\n", data_read_time, global_totaltime - data_read_time, global_totaltime);
         fclose(output);
     }
 
     /* Clean-up allocated memory */
     free(local_data);
     for (int i = 0; i < (local_nx + 2); i++)
     {
         for (int j = 0; j < (local_ny + 2); j++)
         {
             for (int k = 0; k < (local_nz + 2); k++)
             {
                 free(halo[i][j][k]);
             }
             free(halo[i][j]);
         }
         free(halo[i]);
     }
     free(halo);
     for (int i = 0; i < NC; i++) {
         free(local_minima[i]);
         free(local_maxima[i]);
     }
     free(local_minima);
     free(local_maxima);
     free(min_count);
     free(max_count);
     free(rank_minima);
     free(rank_maxima);
     free(global_minima);
     free(global_maxima);
     free(total_local_minima);
     free(total_local_maxima);
 
     MPI_Finalize();
     return 0;
 }
 
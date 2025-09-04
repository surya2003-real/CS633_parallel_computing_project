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
 * exchange_halo: Performs halo exchange among neighboring processes.
 *   - "halo" is a 4D array with extra layers (ghost cells) added in all directions.
 *   - Local dimensions are provided by local_nx, local_ny, and local_nz.
 *   - Process grid dimensions and rank are used to determine neighbors.
 */
void exchange_halo(float ****halo, int local_nx, int local_ny, int local_nz, int NC,
                   int rank, int PX, int PY, int PZ, MPI_Comm comm)
{
    // Determine process coordinates in the 3D grid based on rank.
    int px = rank % PX;
    int py = (rank / PX) % PY;
    int pz = rank / (PX * PY);

    // Determine neighbor ranks in six directions.
    // If the process is on the edge, set to MPI_PROC_NULL.
    int left = (px > 0) ? rank - 1 : MPI_PROC_NULL;
    int right = (px < PX - 1) ? rank + 1 : MPI_PROC_NULL;
    int top = (py > 0) ? rank - PX : MPI_PROC_NULL;
    int bottom = (py < PY - 1) ? rank + PX : MPI_PROC_NULL;
    int front = (pz > 0) ? rank - PX * PY : MPI_PROC_NULL;
    int back = (pz < PZ - 1) ? rank + PX * PY : MPI_PROC_NULL;
    MPI_Request reqs[12];
    int req_count = 0;

    /* Allocate 12 buffers (6 send, 6 receive) */
    float *send_left = (float *)malloc(local_ny * local_nz * NC * sizeof(float));
    float *recv_left = (float *)malloc(local_ny * local_nz * NC * sizeof(float));
    float *send_right = (float *)malloc(local_ny * local_nz * NC * sizeof(float));
    float *recv_right = (float *)malloc(local_ny * local_nz * NC * sizeof(float));

    float *send_front = (float *)malloc(local_nx * local_ny * NC * sizeof(float));
    float *recv_front = (float *)malloc(local_nx * local_ny * NC * sizeof(float));
    float *send_back = (float *)malloc(local_nx * local_ny * NC * sizeof(float));
    float *recv_back = (float *)malloc(local_nx * local_ny * NC * sizeof(float));

    float *send_bottom = (float *)malloc(local_nx * local_nz * NC * sizeof(float));
    float *recv_bottom = (float *)malloc(local_nx * local_nz * NC * sizeof(float));
    float *send_top = (float *)malloc(local_nx * local_nz * NC * sizeof(float));
    float *recv_top = (float *)malloc(local_nx * local_nz * NC * sizeof(float));

    /* Exchange halo data with left neighbor */
    if (left != MPI_PROC_NULL)
    {
        for (int y = 1; y <= local_ny; y++)
            for (int z = 1; z <= local_nz; z++)
                for (int t = 0; t < NC; t++)
                    send_left[(y - 1) * local_nz * NC + (z - 1) * NC + t] = halo[1][y][z][t];

        MPI_Isend(send_left, local_ny * local_nz * NC, MPI_FLOAT, left, 0, comm, &reqs[req_count++]);
        MPI_Irecv(recv_left, local_ny * local_nz * NC, MPI_FLOAT, left, 1, comm, &reqs[req_count++]);
        MPI_Wait(&reqs[req_count - 1], MPI_STATUS_IGNORE);
        MPI_Wait(&reqs[req_count - 2], MPI_STATUS_IGNORE);
        for (int y = 1; y <= local_ny; y++)
            for (int z = 1; z <= local_nz; z++)
                for (int t = 0; t < NC; t++)
                    halo[0][y][z][t] = recv_left[(y - 1) * local_nz * NC + (z - 1) * NC + t];

    }

    /* Exchange halo data with right neighbor */
    req_count = 0;
    if (right != MPI_PROC_NULL)
    {
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

    /* Exchange halo data with front neighbor */
    req_count = 0;
    if (front != MPI_PROC_NULL)
    {
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

    /* Exchange halo data with back neighbor */
    req_count = 0;
    if (back != MPI_PROC_NULL)
    {
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

    /* Exchange halo data with top neighbor */
    req_count = 0;
    if (top != MPI_PROC_NULL)
    {
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

    /* Exchange halo data with bottom neighbor */
    req_count = 0;
    if (bottom != MPI_PROC_NULL)
    {
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


    /* Free all buffers */
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
 * find_local_minima: Finds local minima in the local grid (excluding halo boundary)
 *   - Iterates over all grid points and compares to neighbors (including halo if at the domain boundary).
 *   - For each channel, local minima values are stored in local_minima and counts in min_count.
 */
void find_local_minima(int rank, int PX, int PY, int PZ, float ****halo,
                       int local_nx, int local_ny, int local_nz,
                       float **local_minima, int *min_count)
{

    // Determine process grid positions
    int px = rank % PX;
    int py = (rank / PX) % PY;
    int pz = rank / (PX * PY);

    // Determine if neighboring processes exist
    int left = (px > 0) ? rank - 1 : MPI_PROC_NULL;
    int right = (px < PX - 1) ? rank + 1 : MPI_PROC_NULL;
    int top = (py > 0) ? rank - PX : MPI_PROC_NULL;
    int bottom = (py < PY - 1) ? rank + PX : MPI_PROC_NULL;
    int front = (pz > 0) ? rank - PX * PY : MPI_PROC_NULL;
    int back = (pz < PZ - 1) ? rank + PX * PY : MPI_PROC_NULL;

    // Iterate over local grid points (excluding halo)
    for (int i = 1; i <= local_nx; i++)
    {
        for (int j = 1; j <= local_ny; j++)
        {
            for (int k = 1; k <= local_nz; k++)
            {
                for (int t = 0; t < NC; t++)
                {
                    float val = halo[i][j][k][t]; // Current value

                    int is_min = 1; // Assume it's a minimum initially

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

                    // Check halo regions
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
                    if (is_min) // minimum found
                    {
                        local_minima[t][min_count[t]] = val;
                        (min_count[t])++;
                    }
                }
            }
        }
    }
}

/*
 * find_local_maxima: Finds local maxima in the local grid (excluding halo boundary).
 *   - Each channel's maxima are stored in local_maxima and counts in max_count.
 */
void find_local_maxima(int rank, int PX, int PY, int PZ, float ****halo,
                       int local_nx, int local_ny, int local_nz,
                       float **local_maxima, int *max_count)
{

    // Determine process grid positions
    int px = rank % PX;
    int py = (rank / PX) % PY;
    int pz = rank / (PX * PY);

    // Determine if neighboring processes exist
    int left = (px > 0) ? rank - 1 : MPI_PROC_NULL;
    int right = (px < PX - 1) ? rank + 1 : MPI_PROC_NULL;
    int top = (py > 0) ? rank - PX : MPI_PROC_NULL;
    int bottom = (py < PY - 1) ? rank + PX : MPI_PROC_NULL;
    int front = (pz > 0) ? rank - PX * PY : MPI_PROC_NULL;
    int back = (pz < PZ - 1) ? rank + PX * PY : MPI_PROC_NULL;

    // Iterate over local grid points (excluding halo)
    for (int i = 1; i <= local_nx; i++)
    {
        for (int j = 1; j <= local_ny; j++)
        {
            for (int k = 1; k <= local_nz; k++)
            {
                for (int t = 0; t < NC; t++)
                {
                    float val = halo[i][j][k][t]; // Current value

                    int is_max = 1; // Assume it's a maximum initially

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
                    if (is_max) // maximum found
                    {
                        local_maxima[t][max_count[t]] = val;
                        (max_count[t])++;
                    }
                }
            }
        }
    }
}



int main(int argc, char **argv)
{

    MPI_Init(&argc, &argv);

    int rank, num_procs;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    if (argc != 10)
    {
        // contingency for incorrect number of arguments
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

    // determine local dimensions and calculate rank coordinates
    int local_nx = NX / PX, local_ny = NY / PY, local_nz = NZ / PZ;
    int px = rank % PX;
    int py = (rank / PX) % PY;
    int pz = rank / (PX * PY);

    int right = (px < PX - 1) ? rank + 1 : MPI_PROC_NULL;
    int bottom = (py < PY - 1) ? rank + PX : MPI_PROC_NULL;
    int back = (pz < PZ - 1) ? rank + PX * PY : MPI_PROC_NULL;

    /* Adjust edge subdomains if dimensions are not evenly divisible */
    if (right == MPI_PROC_NULL && (NX % PX))
        local_nx += NX % PX;
    if (bottom == MPI_PROC_NULL && (NY % PY))
        local_ny += NY % PY;
    if (back == MPI_PROC_NULL && (NZ % PZ))
        local_nz += NZ % PZ;
    
    // start time
    double startime, endtime;
    startime = MPI_Wtime();
    float *data = NULL;
    double data_read_time;
    float *local_data = (float *)malloc(local_nx * local_ny * local_nz * NC * sizeof(float));
    data_read_time = MPI_Wtime();
    MPI_File fh;
    MPI_Status status;
    MPI_File_open(MPI_COMM_WORLD, input_file, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);

    int gx = px * local_nx; // global x-start
    int gy = py * local_ny; // global y-start
    int gz = pz * local_nz; // global z-start

    int idx = 0;

    for (int z = 0; z < local_nz; z++) {
        for (int y = 0; y < local_ny; y++) {
            // Compute the offset in terms of floats
            MPI_Offset global_index = (
                ((MPI_Offset)(gz + z) * NY * NX * NC) +
                ((MPI_Offset)(gy + y) * NX * NC) +
                ((MPI_Offset)(gx) * NC)
            );

            MPI_Offset byte_offset = global_index * sizeof(float);

            // Read local_nx * NC floats (a full row of X + channels)
            MPI_File_read_at(fh, byte_offset, &local_data[idx], local_nx * NC, MPI_FLOAT, &status);

            idx += local_nx * NC;
        }
    }

    MPI_File_close(&fh);

    data_read_time = MPI_Wtime() - data_read_time; // Time taken to read data by each rank

    double max_data_read_time;
    MPI_Reduce(&data_read_time, &max_data_read_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD); //reduce to find max time

    // Allocate the 4D array with halo regions (size + 2 in each spatial dimension)
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

    //unpack the local data into the halo regions
    int position = 0;
    for (int z = 1; z <= local_nz; z++)
        for (int y = 1; y <= local_ny; y++)
            for (int x = 1; x <= local_nx; x++)
                for (int t = 0; t < NC; t++)
                    MPI_Unpack(local_data, local_nx * local_ny * local_nz * NC, &position, &halo[x][y][z][t], 1, MPI_FLOAT, MPI_COMM_WORLD);

    // Perform Halo Exchange
    exchange_halo(halo, local_nx, local_ny, local_nz, NC, rank, PX, PY, PZ, MPI_COMM_WORLD);

    /* Find local minima and maxima from the halo data */
    float **local_minima = (float **)malloc(NC * sizeof(float *));
    for (int i = 0; i < NC; i++)
    {
        local_minima[i] = (float *)malloc(local_nx * local_ny * local_nz * sizeof(float));
    }
    int *min_count = (int *)malloc(NC * sizeof(int));
    for (int i = 0; i < NC; i++)
    {
        min_count[i] = 0;
    }
    find_local_minima(rank, PX, PY, PZ, halo, local_nx, local_ny, local_nz, local_minima, min_count);

    float **local_maxima = (float **)malloc(NC * sizeof(float *));
    for (int i = 0; i < NC; i++)
    {
        local_maxima[i] = (float *)malloc(local_nx * local_ny * local_nz * sizeof(float));
    }
    int *max_count = (int *)malloc(NC * sizeof(int));
    for (int i = 0; i < NC; i++)
    {
        max_count[i] = 0;
    }
    find_local_maxima(rank, PX, PY, PZ, halo, local_nx, local_ny, local_nz, local_maxima, max_count);

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
            {
                rank_minima[i] = local_minima[i][j];
            }
        }
        for (int j = 0; j < max_count[i]; j++)
        {
            if (local_maxima[i][j] > rank_maxima[i])
            {
                rank_maxima[i] = local_maxima[i][j];
            }
        }
    }

    // Perform reductions to get global results
    float *global_minima = (float *)malloc(NC * sizeof(float));
    MPI_Reduce(rank_minima, global_minima, NC, MPI_FLOAT, MPI_MIN, 0, MPI_COMM_WORLD);
    float *global_maxima = (float *)malloc(NC * sizeof(float));
    MPI_Reduce(rank_maxima, global_maxima, NC, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);
    int *total_local_minima = (int *)malloc(NC * sizeof(int));
    MPI_Reduce(min_count, total_local_minima, NC, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    int *total_local_maxima = (int *)malloc(NC * sizeof(int));
    MPI_Reduce(max_count, total_local_maxima, NC, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);


    if (rank == 0)
    {
        free(data);
    }
    free(local_data);
    endtime = MPI_Wtime();
    double ttime = endtime - startime; // total time taken by this rank
    double totaltime;
    MPI_Reduce(&ttime, &totaltime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD); // Reduce the local times to find the maximum execution time

    /* Write output to file */
    if (rank == 0)
    {
        FILE *output = fopen(output_file, "w");
        if (!output)
        {
            perror("Error opening output file");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        for (int i = 0; i < NC; i++)
        {
            fprintf(output, "(%d, %d), ", total_local_minima[i], total_local_maxima[i]);
        }
        fprintf(output, "\n");
        for (int i = 0; i < NC; i++)
        {
            fprintf(output, "(%.4f, %.4f), ", global_minima[i], global_maxima[i]);
        }
        fprintf(output, "\n");
        fprintf(output, "%.4f, %.4f, %f\n", data_read_time, totaltime - data_read_time, totaltime);
        fclose(output);
    }

    MPI_Finalize();
    return 0;
}

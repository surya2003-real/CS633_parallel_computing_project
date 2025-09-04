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
    int left   = (px > 0)      ? rank - 1          : MPI_PROC_NULL;
    int right  = (px < PX - 1) ? rank + 1          : MPI_PROC_NULL;
    int top    = (py > 0)      ? rank - PX         : MPI_PROC_NULL;
    int bottom = (py < PY - 1) ? rank + PX         : MPI_PROC_NULL;
    int front  = (pz > 0)      ? rank - PX * PY    : MPI_PROC_NULL;
    int back   = (pz < PZ - 1) ? rank + PX * PY    : MPI_PROC_NULL;

    MPI_Request reqs[12];
    int req_count = 0;

    /* Allocate communication buffers for each direction */
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

    /* Exchange halo data with left neighbor */
    if (left != MPI_PROC_NULL)
    {
        for (int y = 1; y <= local_ny; y++)
        {
            for (int z = 1; z <= local_nz; z++)
            {
                for (int t = 0; t < NC; t++)
                {
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

    /* Exchange halo data with right neighbor */
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

    /* Free all communication buffers */
    free(send_left); free(recv_left);
    free(send_right); free(recv_right);
    free(send_front); free(recv_front);
    free(send_back); free(recv_back);
    free(send_bottom); free(recv_bottom);
    free(send_top); free(recv_top);
}

/*
 * read_data_from_file_subcomm: Uses MPI-IO to collectively read a pz slice.
 *   - All processes in the subcommunicator participate.
 *   - local_nz may be adjusted on the last pz subcommunicator if NZ is not even.
 *   - The file is assumed to be stored as a contiguous array of floats in [z][y][x][c] order.
 */
void read_data_from_file_subcomm(float **data, const char *filename,
                                 int local_nz, MPI_Comm comm, int pz_coord)
{

    int local_nz_actual = local_nz;
    if (pz_coord == PZ - 1 && (NZ % PZ) != 0)
        local_nz_actual = NZ - (PZ - 1) * local_nz;

    size_t elements_per_slice = (size_t)NX * NY * local_nz_actual * NC;
    *data = (float *)malloc(elements_per_slice * sizeof(float));
    if (*data == NULL)
    {
        fprintf(stderr, "Unable to allocate memory for subdomain data on pz = %d\n", pz_coord);
        MPI_Abort(comm, 1);
    }

    MPI_File fh;
    int rc = MPI_File_open(comm, filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
    if (rc != MPI_SUCCESS)
    {
        fprintf(stderr, "Error opening file %s with MPI_File_open\n", filename);
        MPI_Abort(comm, 1);
    }

    MPI_Offset offset = (MPI_Offset)pz_coord * local_nz * NX * NY * NC * sizeof(float);
    rc = MPI_File_set_view(fh, offset, MPI_FLOAT, MPI_FLOAT, "native", MPI_INFO_NULL);
    if (rc != MPI_SUCCESS)
    {
        fprintf(stderr, "Error setting file view in MPI_File_set_view\n");
        MPI_Abort(comm, 1);
    }

    rc = MPI_File_read(fh, *data, elements_per_slice, MPI_FLOAT, MPI_STATUS_IGNORE);
    if (rc != MPI_SUCCESS)
    {
        
        fprintf(stderr, "Error during MPI_File_read_all for pz = %d\n", pz_coord);
        MPI_Abort(comm, 1);
    }

    MPI_File_close(&fh);

    // printf("MPI-IO: Successfully read %zu elements from file for pz = %d\n", elements_per_slice, pz_coord);

}

/*
 * scatter_data_subcomm: Scatters a global pz slice to each process in the subcommunicator.
 *   - global_slice_data is only valid at subrank 0.
 *   - Each process receives its appropriate contiguous block in local_data.
 *   - Uses MPI_Pack_size, MPI_Pack, and MPI_Scatterv for communication.
 */
void scatter_data_subcomm(float *global_slice_data, float *local_data,
                          int local_nx, int local_ny, int local_nz,
                          int num_channels, MPI_Comm comm, int subrank,
                          int subsize)
{

    MPI_Datatype block_type;
    int sizes[4]    = {local_nz, NY, NX, num_channels};
    int subsizes[4] = {local_nz, local_ny, local_nx, num_channels};
    int starts[4]   = {0, 0, 0, 0};

    MPI_Type_create_subarray(4, sizes, subsizes, starts, MPI_ORDER_C, MPI_FLOAT, &block_type);
    MPI_Type_commit(&block_type);

    int pack_size;
    MPI_Pack_size(1, block_type, comm, &pack_size);

    char *sendbuf = NULL;
    int *pack_sizes = NULL;
    int total_pack_size = 0;

    if (subrank == 0)
    {
        pack_sizes = (int *)malloc(subsize * sizeof(int));
        int position = 0;
        for (int p = 0; p < subsize; p++)
        {
            int rank_x = p % PX;
            int rank_y = p / PX;
            int start_x = rank_x * local_nx;
            int start_y = rank_y * local_ny;
            int local_block_dims[4];
            local_block_dims[0] = local_nz;
            local_block_dims[2] = (rank_x == PX - 1 && (NX % PX)) ? NX - start_x : local_nx;
            local_block_dims[1] = (rank_y == PY - 1 && (NY % PY)) ? NY - start_y : local_ny;
            local_block_dims[3] = num_channels;

            MPI_Datatype proc_block_type;
            int global_sizes[4] = {local_nz, NY, NX, num_channels};
            int proc_starts[4]  = {0, start_y, start_x, 0};
            MPI_Type_create_subarray(4, global_sizes, local_block_dims, proc_starts,
                                       MPI_ORDER_C, MPI_FLOAT, &proc_block_type);
            MPI_Type_commit(&proc_block_type);

            MPI_Pack_size(1, proc_block_type, comm, &pack_sizes[p]);
            total_pack_size += pack_sizes[p];

            MPI_Type_free(&proc_block_type);
        }

        sendbuf = (char *)malloc(total_pack_size);
        if (!sendbuf)
        {
            fprintf(stderr, "Unable to allocate send buffer in scatter_data_subcomm\n");
            MPI_Abort(comm, 1);
        }

        int pos = 0;
        for (int p = 0; p < subsize; p++)
        {
            int rank_x = p % PX;
            int rank_y = p / PX;
            int start_x = rank_x * local_nx;
            int start_y = rank_y * local_ny;
            int local_block_dims[4];
            local_block_dims[0] = local_nz;
            local_block_dims[2] = (rank_x == PX - 1 && (NX % PX)) ? NX - start_x : local_nx;
            local_block_dims[1] = (rank_y == PY - 1 && (NY % PY)) ? NY - start_y : local_ny;
            local_block_dims[3] = num_channels;

            MPI_Datatype proc_block_type;
            int global_sizes[4] = {local_nz, NY, NX, num_channels};
            int proc_starts[4]  = {0, start_y, start_x, 0};
            MPI_Type_create_subarray(4, global_sizes, local_block_dims, proc_starts,
                                       MPI_ORDER_C, MPI_FLOAT, &proc_block_type);
            MPI_Type_commit(&proc_block_type);

            MPI_Pack(global_slice_data, 1, proc_block_type, sendbuf,
                     total_pack_size, &pos, comm);

            MPI_Type_free(&proc_block_type);
        }

        free(pack_sizes);
    }

    char *recvbuf = (char *)malloc(pack_size);
    if (!recvbuf)
    {
        fprintf(stderr, "Unable to allocate recv buffer in scatter_data_subcomm\n");
        MPI_Abort(comm, 1);
    }

    int *sendcounts = NULL;
    int *displs = NULL;
    if (subrank == 0)
    {
        sendcounts = (int *)malloc(subsize * sizeof(int));
        displs = (int *)malloc(subsize * sizeof(int));
        int pos = 0;
        for (int p = 0; p < subsize; p++)
        {
            sendcounts[p] = pack_size; // All blocks are packed in same order.
            displs[p] = (p == 0) ? 0 : displs[p - 1] + sendcounts[p - 1];
        }
    }

    MPI_Scatterv(sendbuf, sendcounts, displs, MPI_PACKED,
                 recvbuf, pack_size, MPI_PACKED, 0, comm);

    int pos = 0;
    MPI_Unpack(recvbuf, pack_size, &pos, local_data,
               local_nx * local_ny * local_nz * num_channels, MPI_FLOAT, comm);

    if (subrank == 0)
    {
        free(sendbuf);
        free(sendcounts);
        free(displs);
    }
    free(recvbuf);
    MPI_Type_free(&block_type);
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
    int px = rank % PX;
    int py = (rank / PX) % PY;
    int pz = rank / (PX * PY);

    int left = (px > 0) ? rank - 1 : MPI_PROC_NULL;
    int right = (px < PX - 1) ? rank + 1 : MPI_PROC_NULL;
    int top = (py > 0) ? rank - PX : MPI_PROC_NULL;
    int bottom = (py < PY - 1) ? rank + PX : MPI_PROC_NULL;
    int front = (pz > 0) ? rank - PX * PY : MPI_PROC_NULL;
    int back = (pz < PZ - 1) ? rank + PX * PY : MPI_PROC_NULL;

    for (int i = 1; i <= local_nx; i++)
    {
        for (int j = 1; j <= local_ny; j++)
        {
            for (int k = 1; k <= local_nz; k++)
            {
                for (int t = 0; t < NC; t++)
                {
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
                    if (is_min)
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
    int px = rank % PX;
    int py = (rank / PX) % PY;
    int pz = rank / (PX * PY);

    int left = (px > 0) ? rank - 1 : MPI_PROC_NULL;
    int right = (px < PX - 1) ? rank + 1 : MPI_PROC_NULL;
    int top = (py > 0) ? rank - PX : MPI_PROC_NULL;
    int bottom = (py < PY - 1) ? rank + PX : MPI_PROC_NULL;
    int front = (pz > 0) ? rank - PX * PY : MPI_PROC_NULL;
    int back = (pz < PZ - 1) ? rank + PX * PY : MPI_PROC_NULL;

    for (int i = 1; i <= local_nx; i++)
    {
        for (int j = 1; j <= local_ny; j++)
        {
            for (int k = 1; k <= local_nz; k++)
            {
                for (int t = 0; t < NC; t++)
                {
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
                    if (is_max)
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
        if (rank == 0)
            fprintf(stderr, "Usage: %s <input_file> PX PY PZ NX NY NZ NC <output_file>\n", argv[0]);
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

    int pz_coord = rank / (PX * PY);
    MPI_Comm subcomm;
    MPI_Comm_split(MPI_COMM_WORLD, pz_coord, rank, &subcomm);
    int subrank, subsize;
    MPI_Comm_rank(subcomm, &subrank);
    MPI_Comm_size(subcomm, &subsize);

    MPI_Comm leader_comm;
    int leader_color = (subrank == 0) ? 0 : MPI_UNDEFINED;
    MPI_Comm_split(MPI_COMM_WORLD, leader_color, rank, &leader_comm);
    int leader_rank = -1, leader_size = 0;
    if (subrank == 0)
    {
        MPI_Comm_rank(leader_comm, &leader_rank);
        MPI_Comm_size(leader_comm, &leader_size);
        // printf("Global leader: global rank %d, leader_comm rank %d out of %d leaders\n",
            //    rank, leader_rank, leader_size);
    }
    double startime, endtime;
    startime = MPI_Wtime();
    double data_read_time;
    data_read_time = MPI_Wtime();
    
    float *global_slice_data = NULL;  // Data read by subrank 0 in each subcommunicator.
    if(subrank == 0)
    {
        read_data_from_file_subcomm(&global_slice_data, input_file, local_nz, leader_comm, pz_coord);
    }

    float *local_data = (float *)malloc(local_nx * local_ny * local_nz * NC * sizeof(float));
    scatter_data_subcomm(global_slice_data, local_data, local_nx, local_ny, local_nz, NC, subcomm, subrank, subsize);

    data_read_time = MPI_Wtime() - data_read_time;
    double final_read_time;
    MPI_Reduce(&data_read_time, &final_read_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    /* Unpack the received local data into a 4D halo array with extra boundaries.
       The halo array dimensions are (local_nx+2) x (local_ny+2) x (local_nz+2) x NC.
    */
    float ****halo = (float ****)malloc((local_nx + 2) * sizeof(float ***));
    for (int i = 0; i < (local_nx + 2); i++)
    {
        halo[i] = (float ***)malloc((local_ny + 2) * sizeof(float **));
        for (int j = 0; j < (local_ny + 2); j++)
        {
            halo[i][j] = (float **)malloc((local_nz + 2) * sizeof(float *));
            for (int k = 0; k < (local_nz + 2); k++)
                halo[i][j][k] = (float *)malloc(NC * sizeof(float));
        }
    }

    int position = 0;
    for (int z = 1; z <= local_nz; z++)
        for (int y = 1; y <= local_ny; y++)
            for (int x = 1; x <= local_nx; x++)
                for (int t = 0; t < NC; t++)
                    MPI_Unpack(local_data, local_nx * local_ny * local_nz * NC, &position, &halo[x][y][z][t], 1, MPI_FLOAT, MPI_COMM_WORLD);

    /* Perform halo exchange among the processes (using MPI_COMM_WORLD) */
    exchange_halo(halo, local_nx, local_ny, local_nz, NC, rank, PX, PY, PZ, MPI_COMM_WORLD);

    /* Find local minima and maxima from the halo data */
    float **local_minima = (float **)malloc(NC * sizeof(float *));
    for (int i = 0; i < NC; i++)
        local_minima[i] = (float *)malloc(local_nx * local_ny * local_nz * sizeof(float));
    int *min_count = (int *)malloc(NC * sizeof(int));
    for (int i = 0; i < NC; i++)
        min_count[i] = 0;
    find_local_minima(rank, PX, PY, PZ, halo, local_nx, local_ny, local_nz, local_minima, min_count);
    // for (int i = 0; i < NC; i++)
    //     printf("Rank: %d, Channel: %d, Minima Count: %d\n", rank, i, min_count[i]);

    float **local_maxima = (float **)malloc(NC * sizeof(float *));
    for (int i = 0; i < NC; i++)
        local_maxima[i] = (float *)malloc(local_nx * local_ny * local_nz * sizeof(float));
    int *max_count = (int *)malloc(NC * sizeof(int));
    for (int i = 0; i < NC; i++)
        max_count[i] = 0;
    find_local_maxima(rank, PX, PY, PZ, halo, local_nx, local_ny, local_nz, local_maxima, max_count);
    // for (int i = 0; i < NC; i++)
    //     printf("Rank: %d, Channel: %d, Maxima Count: %d\n", rank, i, max_count[i]);

    /* Compute local and global minima/maxima */
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
            if (local_minima[i][j] < rank_minima[i])
                rank_minima[i] = local_minima[i][j];
        for (int j = 0; j < max_count[i]; j++)
            if (local_maxima[i][j] > rank_maxima[i])
                rank_maxima[i] = local_maxima[i][j];
    }
    float *global_minima = (float *)malloc(NC * sizeof(float));
    MPI_Reduce(rank_minima, global_minima, NC, MPI_FLOAT, MPI_MIN, 0, MPI_COMM_WORLD);
    // if (rank == 0)
    // {
    //     for (int i = 0; i < NC; i++)
    //         printf("Global Minima for Channel %d: %f\n", i, global_minima[i]);
    // }
    float *global_maxima = (float *)malloc(NC * sizeof(float));
    MPI_Reduce(rank_maxima, global_maxima, NC, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);
    // if (rank == 0)
    // {
    //     for (int i = 0; i < NC; i++)
    //         printf("Global Maxima for Channel %d: %f\n", i, global_maxima[i]);
    // }
    int *total_local_minima = (int *)malloc(NC * sizeof(int));
    MPI_Reduce(min_count, total_local_minima, NC, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    // if (rank == 0)
    // {
    //     for (int i = 0; i < NC; i++)
    //         printf("Total Minima for Channel %d: %d\n", i, total_local_minima[i]);
    // }
    int *total_local_maxima = (int *)malloc(NC * sizeof(int));
    MPI_Reduce(max_count, total_local_maxima, NC, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    // if (rank == 0)
    // {
    //     for (int i = 0; i < NC; i++)
    //         printf("Total Maxima for Channel %d: %d\n", i, total_local_maxima[i]);
    // }

    if (subrank == 0)
        free(global_slice_data);
    free(local_data);

    endtime = MPI_Wtime();
    double ttime = endtime - startime;
    double totaltime;
    MPI_Reduce(&ttime, &totaltime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    // if (rank == 0)
    // {
    //     printf("Total Time: %f\n", totaltime);
    //     printf("Data Read Time: %f\n", final_read_time);
    //     printf("Main Program Time: %f\n", totaltime - final_read_time);
    // }

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
            fprintf(output, "(%d, %d), ", total_local_minima[i], total_local_maxima[i]);
        fprintf(output, "\n");
        for (int i = 0; i < NC; i++)
            fprintf(output, "(%.4f, %.4f), ", global_minima[i], global_maxima[i]);
        fprintf(output, "\n");
        fprintf(output, "%f, %f, %f\n", final_read_time, totaltime - final_read_time, totaltime);
        fclose(output);
    }

    // printf("Rank %d of %d finalizing...\n", rank, num_procs);
    MPI_Finalize();
    return 0;
}

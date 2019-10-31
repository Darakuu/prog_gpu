int A[3][4]; // A[r][c]

int **C; // criminale
int *G;

int main()
{
  int nrows, ncols;
/* Peggiore
  C = malloc(nrows*sizeof(int*));
  for (int r = 0; r < nrows; ++r)
    C[r] = malloc(ncols*sizeof(int));
*/
/*
  Meno peggio
  C = malloc(nrows*sizeof(int*));
  C[0] = malloc(nrows*ncols*sizeof(int));
  for (int r = 1; r < nrows; ++r)
    C[r] = C[0] + r*ncols;
*/
  // "Modo giusto"
  G = malloc(nrows*ncols*sizeof(int));
  //G[r*ncols+c]

}
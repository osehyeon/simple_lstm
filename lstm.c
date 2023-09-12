static inline void node_lstm0( const float X[10][1][1], const float W[1][128][1], const float R[1][128][32], const float B[1][256], float Y[10][1][1][32], float Y_h[1][1][32], float Y_c[1][1][32] )
{
   /* LSTM 
    * inputs: 
    *   X = tensor_input
    *   W = tensor_W0w
    *   R = tensor_R0
    *   B = tensor_B0
    *   sequence_lens = 
    *   initial_h = 
    *   initial_c = 
    *   P = 
    * outputs: 
    *   Y = tensor_layer_output_0
    *   Y_h = tensor_last_state_0
    *   Y_c = tensor_last_cell_state_0
    * attributes:
    *   activations: Sigmoid Tanh Tanh 
    * clip: off
    * layout: 0
    * (rest TBD):
    */
   int hs = 32;
   int ds = 1;
   int bs = 1;
   int iidx = 0;
   int oidx = hs;
   int fidx = 2*hs;
   int cidx = 3*hs;
   int Rb = 4*hs;
   int sequence_lenght = 10;
   /* Forget gate */
   float ft[bs][hs];
   /* Input gate */
   float it[bs][hs];
   /* Cell gate */
   float ct[bs][hs];
   /* Output gate */
   float ot[bs][hs];

   memset(Y_h, 0, sizeof(*Y_h));
   memset(Y_c, 0, sizeof(*Y_c));

   for( int s=0; s<sequence_lenght; s++) {

      /* Forward lane */
      for( int b=0; b<bs; b++)
      for( int h=0; h<hs; h++) {
         ft[b][h]=0;
         it[b][h]=0;
         ct[b][h]=0;
         for( int i=0; i<ds; i++) {
            ft[b][h] += X[s][b][i]*W[0][fidx+h][i];
            it[b][h] += X[s][b][i]*W[0][iidx+h][i];
            ct[b][h] += X[s][b][i]*W[0][cidx+h][i];
         }
         for( int k=0; k<hs; k++) {
            ft[b][h] += Y_h[0][b][k]*R[0][fidx+h][k];
            ct[b][h] += Y_h[0][b][k]*R[0][cidx+h][k];
            it[b][h] += Y_h[0][b][k]*R[0][iidx+h][k];
         }
         ft[b][h] += B[0][fidx+h];
         ft[b][h] += B[0][Rb+fidx+h];
         it[b][h] += B[0][iidx+h];
         it[b][h] += B[0][Rb+iidx+h];
         ct[b][h] += B[0][cidx+h];
         ct[b][h] += B[0][Rb+cidx+h];
         ft[b][h] =1.0f/(1+expf(-ft[b][h]));
         it[b][h] =1.0f/(1+expf(-it[b][h]));
         ct[b][h] =tanh(ct[b][h]);
      }
      for( int b=0; b<bs; b++)
      for( int h=0; h<hs; h++) {
         /* Cell state */
         Y_c[0][b][h] = Y_c[0][b][h]*ft[b][h] + it[b][h]*ct[b][h];
         /* Output gate */
         ot[b][h]=0;
         for( int i=0; i<ds; i++)
            ot[b][h] += X[s][b][i]*W[0][oidx+h][i];
         for( int k=0; k<hs; k++)
            ot[b][h] += Y_h[0][b][k]*R[0][oidx+h][k];
         ot[b][h] += B[0][oidx+h];
         ot[b][h] += B[0][Rb+oidx+h];
         ot[b][h] =1.0f/(1+expf(-ot[b][h]));
      }
      /* Hidden state */
      for( int b=0; b<bs; b++)
      for( int h=0; h<hs; h++) {
         Y_h[0][b][h] = ot[b][h] * tanh(Y_c[0][b][h]);
         Y[s][0][b][h]= Y_h[0][b][h];
      }

   } /* sequences */
}

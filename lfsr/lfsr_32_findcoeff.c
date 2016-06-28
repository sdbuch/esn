// Given data from source, find good polynomial for LFSR
// "Good" is a polynomial having minimum taps and taps close
// together.
// Source is:
// http://www.ece.cmu.edu/~koopman/lfsr/index.html -> 32.dat
//


#include <stdlib.h>
#include <stdio.h>

const int BITS_IN_LFSR = 32;
unsigned int set_bits(unsigned int);
float mean_diff(unsigned int);

int main(int argc, char** argv) {
  if(argc < 1 || argc > 2) {
    exit(-1);
  }

  FILE* f;
  int len_line = BITS_IN_LFSR/4+1;
  char word[len_line];
  unsigned int val;
  unsigned int cur_c;
  float cur_d;
  unsigned int champ = 0;
  unsigned int champ_c = BITS_IN_LFSR;
  float champ_d = (float)BITS_IN_LFSR;
  
  f = fopen(argv[1], "r");
  
  if (f != NULL) {
    while ( fgets(word, len_line, f) != NULL) {
      val = (unsigned int)strtol(word,NULL,16);
      //cur_c = set_bits(val);
      cur_c = __builtin_popcount(val);
      if (cur_c <= champ_c) {
        cur_d = mean_diff(val);
        if (cur_c == champ_c) {
          if (cur_d < champ_d) {
            champ = val;
            champ_d = cur_d;
          }
        }
        else {
          champ = val;
          champ_c = cur_c;
          champ_d = cur_d;
        }
      }
      // Get the new line character
      fgets(word, len_line, f);
    }
  }
  else {
    printf("%s", argv[1]);
    perror("File could not be opened");
  }
  
  printf(" The best result was %u, with count %u and diff %f \n", champ, champ_c, champ_d);
  fclose(f);

  return 0;
}

unsigned int set_bits(unsigned int k) {
  unsigned int count=0;
  while (k) {
    k &= (k-1);
    count++;
  }
  return count;

}

float mean_diff(unsigned int k) {
  unsigned int indices[BITS_IN_LFSR];
  unsigned int idx = 0;
  unsigned int N = 0;
  unsigned int sum = 0;

  for (unsigned int i=0; i<BITS_IN_LFSR; i++) {
    if ( (k >> i) & 1) {
      indices[idx] = i;
      N++;
      idx++;
    }
  }

  for (int i=0; i<N-1; i++) {
    sum += (indices[i+1] - indices[i]);
  }

  return (float)sum / (float)(N-1);

}

struct foo {
  unsigned int I:1;
  unsigned char J[1];
  unsigned int K:1;
 };

void test(struct foo *X) {}


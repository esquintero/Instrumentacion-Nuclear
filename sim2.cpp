#include <iostream>
#include <cmath>
#include <random>
#include <vector>
#include <algorithm>

void leyexpo(int N0, double gamma, double tf, double dt, std::vector<double> &Nt);
void grilla(double tf, double dt, std::vector<double> &grid);
void simexpo(double tf, double gamma, std::vector<double> &grid, double dt, std::vector<double> &frecexpo);
void simbin(int N0, double tf, std::vector<double> &frecbin);

const int N0 = 100000000;
const double dt = 0.2;
const double Gamma = 0.7/2.6;
  
int main(void)
{
  double tf = 37/dt;
  std::vector<double> grid(tf+1);
  std::vector<double> Nt(tf+1);
  std::vector<double> frecexpo(tf+1);
  std::vector<double> frecbin(tf+1);

  grilla(tf, dt, grid);
  leyexpo(N0, Gamma, tf, dt, Nt);
  simexpo(tf, Gamma, grid, dt, frecexpo);
  simbin(N0, tf, frecbin);

  double M = *std::max_element(frecexpo.begin(), frecexpo.end());
  for(int kk = 0; kk < tf; ++kk){
    std::cout << grid[kk] << "\t"
	      << Nt[kk]*M/N0 << "\t"
	      << frecexpo[kk] << "\t"
	      << frecbin[kk]*M/N0 << "\n";
      //<< frecexpo[kk] << "\t"
      //      << M << "\n";
  }
  return 0;
}

void leyexpo(int N0, double gamma, double tf, double dt, std::vector<double> &Nt)
{
  for(int ii = 0; ii < tf; ++ii){
    Nt[ii] = N0*exp(-gamma*ii*dt);
  }
}

void grilla(double tf, double dt, std::vector<double> &grid)
{ 
  for(int ii = 0; ii < tf; ++ii){
    grid[ii] = dt*ii;
  }
}

void simexpo(double tf, double gamma, std::vector<double> &grid, double dt, std::vector<double> &frecexpo)
{
  std::vector<double> fdp(tf+1);
  std::vector<double> Facum(tf+1);
  int bingo = 0;
  double r = 0.0;
  int seed = 1;
  std::mt19937 gen(seed);
  std::uniform_real_distribution<> dis(0,1); //dis(gen) para llamar el random

  for(int ii = 0; ii < tf; ++ii){
    fdp[ii] = -std::exp(-gamma*(grid[ii]+dt)) + std::exp(-gamma*grid[ii]);
  }

  for(int ii = 0; ii < tf; ++ii){
    Facum[ii+1] = Facum[ii] + fdp[ii + 1];
  }

  for(int ii = 1; ii <= N0; ++ii){
    r = dis(gen);
    for(int jj = 1; jj < tf; ++jj){
      if(Facum[jj - 1] <= r && r < Facum[jj]){
	bingo = jj;
	break;
      }
    }
    frecexpo[bingo] += 1; 
  }
}

void simbin(int N0, double tf, std::vector<double> &frecbin)
{
  double aux = N0;
  int seed = 1;
  double r = 0.0;
  double p = (0.7/2.6)*0.2;
  double q = 1-p;
  std::mt19937 gen(seed);
  std::uniform_real_distribution<> dis(0,1); //dis(gen) para llamar el random

  for(int ii = 0; ii < tf; ++ii){
    for(int jj = 1; jj <= aux; ++jj){
      r = dis(gen);
      if(r < q){
	frecbin[ii] += 1;
      }
    }
    aux = frecbin[ii];
  }
}

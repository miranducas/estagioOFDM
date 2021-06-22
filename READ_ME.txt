

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Instruções para rodar os scripts:
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
-comandos.sh --> bash script (para LINUX) que roda os comandos a seguir automaticamente para todas as SNRs e plota a curva
de BER no fim.

-Train.py -->  esse script treina os métodos ofdm que utilizam rede neural e salva cada modelo em um arquivo do tipo .h5. O treino é realizado para um valor fixo de SNR. Os modelos salvos são utilizados pelo script Main_ml.py.

-Main_ml.py --> esse script gera BER média para todos os métodos ofdm para um valor de SNR fixo. A BER média de cada método ofdm é salva em arquivo pickle para ser utilizado no script_figura.py

-script_figura.py --> esse script gera gráfico de BER média X SNR para todos os métodos ofdm

-Main_only_non_ml.py --> esse script gera gráfico de BER média X SNR apenas para os métodos ofdm que não usam machine learning.

-ofdm.py --> biblioteca que contém as principais funções utilizados em um sistema ofdm

-OFDM_system --> biblioteca que contém as classes de blocos transmissores e receptores OFDM

-ofdm_methods --> biblioteca que os métodos ofdm organizados em classes.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Aviso: Treinar as redes e avaliar a BER média no mesmo script (em python) pode causar travamento da máquina.
O mesmo pode ocorrer se tentar rodar todas as SNRs de uma vez no mesmo script Main_ml.py. Para contornar este problema,
existe o script em bash (para LINUX) : comandos.sh que só precisa ser rodado uma vez.
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

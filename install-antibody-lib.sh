# Install AbsolutNoLib
wget https://github.com/csi-greifflab/Absolut/archive/a50b3e41e2b7170aee207f067cf8d7009234c30e.zip -P _benchmark_problems/antibody_design//libs/
cd _benchmark_problems/antibody_design/libs/
unzip a50b3e41e2b7170aee207f067cf8d7009234c30e.zip
rm a50b3e41e2b7170aee207f067cf8d7009234c30e.zip
mv Absolut-a50b3e41e2b7170aee207f067cf8d7009234c30e Absolut
cd Absolut/src
make
cd ../../../../..

mkdir -p _benchmark_problems/antibody_design/libs/Absolut/src/antigen_data/2DD8_S
cd _benchmark_problems/antibody_design/libs/Absolut/src/antigen_data/2DD8_S
wget https://ns9999k.webs.sigma2.no/10.11582_2021.00063/projects/NS9603K/pprobert/AbsolutOnline/Structures/SUDDL6142d1af0c3837a24ca534e96b7192bb-10-11-af8a57bbcc249709ce1058ec80ba20a4Structures.txt.zip
unzip SUDDL6142d1af0c3837a24ca534e96b7192bb-10-11-af8a57bbcc249709ce1058ec80ba20a4Structures.txt.zip
rm SUDDL6142d1af0c3837a24ca534e96b7192bb-10-11-af8a57bbcc249709ce1058ec80ba20a4Structures.txt.zip

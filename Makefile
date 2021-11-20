sim2.x: sim2.cpp
	g++ sim2.cpp -o sim2.x
plot: sim2.x
	sim2.x > data.txt
	gnuplot
clean:
	rm -f *.x *.txt


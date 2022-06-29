bld/gemmulem: src/EM.cpp src/main.cpp src/EM.h
	@if [ ! -d "bld" ]; \
	then \
		mkdir bld; \
	fi; \
	g++ -O3 -o gemmulem src/EM.cpp src/main.cpp; \
	mv gemmulem bld

clean: bld 
	@rm -rf bld

install: bld/gemmulem
	@cp bld/gemmulem /bin/

uninstall: /bin/gemmulem
	@rm /bin/gemmulem

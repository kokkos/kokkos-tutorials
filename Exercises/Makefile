all: begin solution

begin:
	make -j 8 -C 01/Begin
	make -j 8 -C 02/Begin
	make -j 8 -C 03/Begin
	make -j 8 -C 04/Begin
	make -j 8 -C 05/Begin
	make -j 8 -C 06/Begin
	make -j 8 -C 07/Begin
	make -j 8 -C 08/Begin
	make -j 8 -C 09/Begin
	make -j 8 -C 10/Begin

solution:
	make -j 8 -C 01/Solution
	make -j 8 -C 02/Solution
	make -j 8 -C 03/Solution
	make -j 8 -C 04/Solution
	make -j 8 -C 05/Solution
	make -j 8 -C 06/Solution
	make -j 8 -C 07/Solution
	make -j 8 -C 08/Solution
	make -j 8 -C 09/Solution
	make -j 8 -C 10/Solution

test: test-begin test-solution

test-begin: begin
	make -j 8 -C 01/Begin test
	make -j 8 -C 02/Begin test
	make -j 8 -C 04/Begin test
	make -j 8 -C 05/Begin test
	make -j 8 -C 06/Begin test
	make -j 8 -C 07/Begin test

test-solution: solution
	make -j 8 -C 01/Solution test
	make -j 8 -C 02/Solution test
	make -j 8 -C 03/Solution test
	make -j 8 -C 04/Solution test
	make -j 8 -C 05/Solution test
	make -j 8 -C 06/Solution test
	make -j 8 -C 07/Solution test

clean:
	make -j 8 -C 01/Begin clean
	make -j 8 -C 01/Solution clean
	make -j 8 -C 02/Begin clean
	make -j 8 -C 02/Solution clean
	make -j 8 -C 03/Begin clean
	make -j 8 -C 03/Solution clean
	make -j 8 -C 04/Begin clean
	make -j 8 -C 04/Solution clean
	make -j 8 -C 05/Begin clean
	make -j 8 -C 05/Solution clean
	make -j 8 -C 06/Begin clean
	make -j 8 -C 06/Solution clean
	make -j 8 -C 07/Begin clean
	make -j 8 -C 07/Solution clean
	make -j 8 -C 08/Begin clean
	make -j 8 -C 08/Solution clean
	make -j 8 -C 09/Begin clean
	make -j 8 -C 09/Solution clean
	make -j 8 -C 10/Begin clean
	make -j 8 -C 10/Solution clean
	make -j 8 -C scatter_view/Begin clean
	make -j 8 -C scatter_view/Solution clean

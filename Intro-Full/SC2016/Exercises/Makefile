all: begin solution

begin:
	make -C 01/Begin
	make -C 02/Begin
	make -C 03/Begin
	make -C 04/Begin
	make -C 05/Begin
	make -C 06/Begin
	make -C 07/Begin

solution:
	make -C 01/Solution
	make -C 02/Solution
	make -C 03/Solution
	make -C 04/Solution
	make -C 05/Solution
	make -C 06/Solution
	make -C 07/Solution

test: test-begin test-solution

test-begin: begin
	make -C 01/Begin test
	make -C 02/Begin test
	make -C 03/Begin test
	make -C 04/Begin test
	make -C 05/Begin test
	make -C 06/Begin test
	make -C 07/Begin test

test-solution: solution
	make -C 01/Solution test
	make -C 02/Solution test
	make -C 03/Solution test
	make -C 04/Solution test
	make -C 05/Solution test
	make -C 06/Solution test
	make -C 07/Solution test

clean:
	make -C 01/Begin clean
	make -C 01/Solution clean
	make -C 02/Begin clean
	make -C 02/Solution clean
	make -C 03/Begin clean
	make -C 03/Solution clean
	make -C 04/Begin clean
	make -C 04/Solution clean
	make -C 05/Begin clean
	make -C 05/Solution clean
	make -C 06/Begin clean
	make -C 06/Solution clean
	make -C 07/Begin clean
	make -C 07/Solution clean

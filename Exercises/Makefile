all: begin solution

begin:
	make -j 8 -C 01/Begin
	make -j 8 -C 02/Begin
	make -j 8 -C 03/Begin
	make -j 8 -C 04/Begin
	make -j 8 -C dualview/Begin
	make -j 8 -C random_number/Begin
	make -j 8 -C scatter_view/Begin
	make -j 8 -C scatter_view/Usage
	make -j 8 -C subview/Begin
	make -j 8 -C tasking/Begin
	make -j 8 -C team_policy/Begin
	make -j 8 -C team_scratch_memory/Begin
	make -j 8 -C team_vector_loop/Begin
	make -j 8 -C unique_token/Begin
	make -j 8 -C unordered_map/Begin
	make -j 8 -C virtualfunction/Begin

solution:
	make -j 8 -C 01/Solution
	make -j 8 -C 02/Solution
	make -j 8 -C 03/Solution
	make -j 8 -C 04/Solution
	make -j 8 -C dualview/Solution
	make -j 8 -C random_number/Solution
	make -j 8 -C scatter_view/Solution
	make -j 8 -C subview/Solution
	make -j 8 -C tasking/Solution
	make -j 8 -C team_policy/Solution
	make -j 8 -C team_scratch_memory/Solution
	make -j 8 -C team_vector_loop/Solution
	make -j 8 -C unique_token/Solution
	make -j 8 -C unordered_map/Solution
	make -j 8 -C virtualfunction/Solution

test: test-begin test-solution

test-begin: begin
	make -j 8 -C 01/Begin test
	make -j 8 -C 02/Begin test
	make -j 8 -C 04/Begin test
	make -j 8 -C dualview/Begin test
	make -j 8 -C random_number/Begin test
	make -j 8 -C scatter_view/Begin test
	make -j 8 -C subview/Begin test
	make -j 8 -C tasking/Begin test
	make -j 8 -C team_policy/Begin test
	make -j 8 -C team_scratch_memory/Begin test
	make -j 8 -C team_vector_loop/Begin test
	make -j 8 -C unique_token/Begin test
	make -j 8 -C unordered_map/Begin test
	make -j 8 -C virtualfunction/Begin test

test-solution: solution
	make -j 8 -C 01/Solution test
	make -j 8 -C 02/Solution test
	make -j 8 -C 03/Solution test
	make -j 8 -C 04/Solution test
	make -j 8 -C dualview/Solution test
	make -j 8 -C random_number/Solution test
	make -j 8 -C scatter_view/Solution test
	make -j 8 -C subview/Solution test
	make -j 8 -C tasking/Solution test
	make -j 8 -C team_policy/Solution test
	make -j 8 -C team_scratch_memory/Solution test
	make -j 8 -C team_vector_loop/Solution test
	make -j 8 -C unique_token/Solution test
	make -j 8 -C unordered_map/Solution test
	make -j 8 -C virtualfunction/Solution test

clean:
	make -j 8 -C 01/Begin clean
	make -j 8 -C 01/Solution clean
	make -j 8 -C 02/Begin clean
	make -j 8 -C 02/Solution clean
	make -j 8 -C 03/Begin clean
	make -j 8 -C 03/Solution clean
	make -j 8 -C 04/Begin clean
	make -j 8 -C 04/Solution clean
	make -j 8 -C dualview/Begin clean
	make -j 8 -C dualview/Solution clean
	make -j 8 -C random_number/Begin clean
	make -j 8 -C random_number/Solution clean
	make -j 8 -C scatter_view/Begin clean
	make -j 8 -C scatter_view/Solution clean
	make -j 8 -C scatter_view/Usage clean
	make -j 8 -C subview/Begin clean
	make -j 8 -C subview/Solution clean
	make -j 8 -C tasking/Begin clean
	make -j 8 -C tasking/Solution clean
	make -j 8 -C team_policy/Begin clean
	make -j 8 -C team_policy/Solution clean
	make -j 8 -C team_scratch_memory/Begin clean
	make -j 8 -C team_scratch_memory/Solution clean
	make -j 8 -C team_vector_loop/Begin clean
	make -j 8 -C team_vector_loop/Solution clean
	make -j 8 -C unique_token/Begin clean
	make -j 8 -C unique_token/Solution clean
	make -j 8 -C unordered_map/Begin clean
	make -j 8 -C unordered_map/Solution clean
	make -j 8 -C virtualfunction/Begin clean
	make -j 8 -C virtualfunction/Solution clean

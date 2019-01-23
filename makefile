help:
	@echo Put this script at the root of the git project.
	@echo
	@echo Add the following lines to the .gitignore file:
	@echo "    ""**/UV/**"
	@echo "    ""!**/UV/unversioned"
	@echo
	@echo Run
	@echo "    ""make unversioned"
	@echo to create the listings of unversioned files in
	@echo "    ""**/UV/unversioned"
	@echo and stage these for commit.

mark_unversioned:
	@echo -----------
	find -name 'UV' -type d | sed 's|.*|&/unversioned|' | xargs -L1 touch
	find -name 'unversioned' | xargs git add
	@echo -----------
	git status

unversioned:
	make mark_unversioned
	
	for f in $$(find -name 'unversioned' -type f); do \
		d=$$(dirname $$f); \
		echo $$d; \
		LANG=US.UTF-8 ls -al $$d | grep -v ' [\.]*[ ]*$$' | cut -d ' ' -f 5- | grep -v ' unversioned[ ]*$$' > $$f; \
		cat $$f; \
	done
	
	git commit -m "(listing unversioned directories)" $$(find -name 'unversioned' -type f)

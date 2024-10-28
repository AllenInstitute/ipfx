### When you are ready to release:
Once the development branch has acquired enough features for a release
or a predetermined release date is approaching

## GitHub

- [ ] Assign a developer to be responsible for the release deployment
- [ ] Create a release branch from dev
- [ ] Create a draft pull request for the release
  - [ ] Add the Project Owner as a reviewer
  - [ ] Copy this checklist into the draft pull request description
- [ ] Prepare the official release commit
  - [ ] Bump version in the version.txt
  - [ ] Move changes from the "Unreleased" section to the proper sections in the CHANGELOG.md
  - [ ] Confirm all GitHub Actions tests pass
  - [ ] Change the draft to pull request to "ready for review"
  - [ ] Code Review with the Project Owner
  - [ ] When it is ready, merge into the master branch; this will generate a merge commit, and this commit will be the official release commit.
- [ ] Create a Release: https://github.com/AllenInstitute/ipfx/releases <"Draft a new release" button>
  - [ ] Create a draft release
  - [ ] Summarize the release notes from the CHANGELOG.md and post them on on the Releases page on GitHub
  - [ ] Review the release with the Project Owner
  - [ ] Publish the release

### PyPI and BKP

- [ ] Reconfirm the version is correct in `version.txt`
- [ ] Set release version with "git tag v#.#.#" (e.g. "git tag v1.0.0", equivalent to the version you bumped to), this triggers circleci to publish ipfx to PyPI (deprecated, need to move to GitHub Actions)
- [ ] After release/deployment, merge master branch (bug fixes, document generation, etc.) back into dev and delete the release branch
- [ ] Build and deploy:
    - [ ] `python setup.py sdist` and `python setup.py bdist_wheel`
    - [ ] `twine upload dist/* --verbose --config-file ~/.pypirc`
- [ ] Announce release on https://community.brain-map.org


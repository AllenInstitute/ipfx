# Contributing to the IPFX

Thank you for your interest in contributing! There are a few ways you can contribute
to the project:

* [Asking/Answering questions on the forurm](#answering-user-questions)
* [Reporting bugs](#reporting-bugs)
* [Requesting features](#suggesting-featuresenhancements)
* [Contributing code](#contributing-code)

## Contributing to community forum
A great way to contribute to the IPFX is by participating in the community discussion on the
the [forums](https://community.brain-map.org). By asking new questions and
answering questions from other users you will help growing the forum as a go-to collaborative scientific resource.

### Have a question? 

Is your question about the IPFX, or about Allen Institute data and tools more generally?
* If you have an IPFX question, first check the [documentation](https://ipfx.readthedocs.io) (including the [gallery of examples](https://ipfx.readthedocs.io/en/latest/auto_examples/index.html) and the [api reference](https://ipfx.readthedocs.io/en/latest/ipfx.html)) to make sure that your question is not already addressed.
    * If you can't find an answer in the documentation, please create an issue on Github.
* If you have question about data and other tools, you should check our [online help](http://help.brain-map.org) or the [Allen Brain Map Community Forum](https://community.brain-map.org). 
    * If you can't find what you are looking for with the aforementioned resources, you can submit your question using [this form](http://allins.convio.net/site/PageServer?pagename=send_us_a_message_ai).


## Bug reports and feature requests

Before reporting a bug or requesting a feature, use Github's issue search to see if anyone else has already done so.

### Reporting bugs 
If there is no existing issue, create a new one. You should include:
* A brief, descriptive title
* A clear description of the problem 
* If you are reporting a bug, your description should contain the following information:
    * What you did (preferably the actual code or commands that you ran)
    * What happened
    * What you were expecting to happen
    * How your system is configured (operating system, Python version)

If you are comfortable addressing this issue yourself, take a look at the [guide to contributing code](#contributing-code) below.


### Suggesting features/enhancements
Before suggesting a feature or enhancement, please check existing issues as you may find out
you don't need to create one. When you create an enhancement suggestion, please include
as many details as possible in the issue template.

When contributing a new feature to the IPFX, the maintenance burden is (by default)
transferred to the IPFX team. This means that the benefit of the contribution must be
weighed against the cost of maintaining the feature. 

When suggesting a feature, consider:
* Is the change clearly explained and motivated?
* Would the enhancement be useful for most users?
* Is this a new feature that can stand alone as a third party project?
* How does this change impact existing users?


## Contributing code

If you are able to improve the IPFX, send us your pull requests!
Contributing code yourself can be a great way to include the features you want
in the IPFX.

### Deciding What to Contribute

Navigate to the Github ["issues"](https://github.com/AllenInstitute/ipfx/issues) 
tab and start looking through
issues. The IPFX team uses Github issues to track our internal development, so we 
recommend filtering to issues with the "good first issue" label or issues with the
"help wanted" label. These are issues that we believe are particularly well
suited for outside contributions, often because we won't get to them right away.
If you decide to start on an issue, leave a comment so that other people know that 
you are working on it.

Code contributions should be submitted in the form of a pull request. Here are the steps:

* Create up an issue/branch/environment
* Sign Contributor License Agreement (CLA)
* Check if changes are consistent with [Coding style](#style-guidelines)
* Commit your code changes
* [Write unit tests](#testing)
* Run unit tests 
* Update the "Unreleased" section in the CHANGELOG.md with the change made in this PR  
* Make a pull request
* Go through the review process

### Setting up

* Make sure that there is an issue tracking your work. See [above](#bug-reports-and-feature-requests) for guidelines on creating effective issues.
* Create a [fork](https://help.github.com/articles/fork-a-repo/) of the IPFX and clone it to your development environment.

* Make a new branch for your code off of `master`. For consistency and use with 
visual git plugins, we prefer the following convention for branch naming:
`GH-<issue-number>/<bugfix/feature>/<short-description>`. For example:
    ```
    GH-712/bugfix/auto-reward-key
    GH-9999/feature/parallel-behavior-analysis
    ```
* Create an environment and install necessary requirements: `requirements.txt` and `requirements-test.txt`
* Start writing code!

### Style guidelines
We follow [PEP-8 guidelines](https://www.python.org/dev/peps/pep-0008/) for new python code.
We also follow [PEP-484](https://www.python.org/dev/peps/pep-0484/) for type annotations.
Before submitting a pull request, run [flake8](https://pypi.python.org/pypi/flake8/) and 
[mypy](https://pypi.org/project/mypy/) linters to check the style of your code. All new code contributions should be compatible with Python 3.6+.

Docstrings for new code should follow the [Numpy docstring standard](https://numpydoc.readthedocs.io/en/latest/format.html). This allows us to ensure consistency in our auto-generated API documentation.

### Committing
Commit messages should have a subject line, separated by a blank line and then 
paragraphs of approximately 72 char lines. For the subject line, shorter is better --
ideally 72 characters maximum. In the body of the commit message, more detail
is better than less. See [Chris Beams](https://chris.beams.io/posts/git-commit/) for
more guidelines about writing good commit messages.

* Tag the issue number in your subject line. For Github issues, it's helpful to 
use the abbreviation ("GH") to separate it from Jira tickets.
    ```
    GH #1111 - Add commit message guidelines

    This contains more detailed information about the feature
    or bugfix. It's written in complete sentences. It has
    appropriate capitalization and punctuation. It's separated
    from the subject by a blank line.
    ```
* Limit commits to the most granular changes that make sense. Group together small
units of work into a single commit when applicable. Think about readability;
your commits should tell a story about your changes that everyone can follow. 

### Testing
All code you write should have unit tests, including bugfixes (since the presence of bugs
likely indicates a gap in test coverage). We use [pytest](https://docs.pytest.org/en/latest/) 
for running unit tests.

If you write a new file `foo.py`, you should place its unit tests in `test_foo.py`.
Follow the directory structure of the parent module(s) for your tests so that 
they are easy to find. For example, tests for `ipfx/dataset/foo.py`
should be in `ipfx/tests/dataset/test_foo.py`.

**Testing guidelines**
* Smaller, faster tests are better (and more likely to be run!)
* Tests should be deterministic
* Tests should be hermetic. They should be packed with everything they need and start any fake services they might need.
* Tests should work every time; use dependency injection to mock out flaky or long-running services.

### Making a pull request
* Make sure your tests pass locally first (`make test` or `python -m pytest <a test file or directory>`)
* Update your forked repository and rebase your branch onto the latest `master` branch.
* Target the latest release candidate branch for your PR. This branch has the format `rc/x.y.z`.
* Use a brief but descriptive title.
* Include `Relates to: #issue_number` and a short description of your changes in the
body of the pull request.
* Support your changes with additional resources. Having an example notebook
or visualizations can be very helpful during the review process.

### Review process
Once your pull request has been made and your tests are passing, a member of the IPFX
team will be assigned to review. Please be patient, as it can take some time to 
assign team members. Once your pull request has been approved, the IPFX
team member will merge your changes into the latest release candidate branch.
Your changes will be included in the next release cycle. Releases typically
occur every 2-4 weeks.

If in doubt how to do anything, don't hesitate to ask a team member!

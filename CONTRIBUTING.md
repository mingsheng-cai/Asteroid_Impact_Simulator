Libraries usually follow the [semantic versioning](https://semver.org/) scheme.

To simplify the usability of this scheme, it is important to adopt some rules. Especially when a diverse set of people are contributing to the same project.

So the following rules are adopted in this project:

| **Type**  | **Description**                                                               |
|-----------|-------------------------------------------------------------------------------|
| feat      | A new feature                                                                 |
| fix       | A bug fix                                                                     |
| docs      | Documentation only changes                                                    |
| style     | Changes that do not affect the meaning of the code (white-space, formatting, missing semi-colons, etc) |
| refactor  | A code change that neither fixes a bug nor adds a feature                     |
| perf      | A code change that improves performance                                       |
| test      | Adding missing tests or correcting existing tests                             |
| chore     | Changes to the build process or auxiliary tools and libraries such as documentation generation |

The following are examples of commit messages that are acceptable:

```
feat: implement the `great_circle_distance` for the `GeospatialLocator` class
fix: fix bug in the distance computation
perf: improve the performance of the `Planet` class 
```

Note: text generated witht he help with Copilot.

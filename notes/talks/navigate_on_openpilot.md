## [Navigate on openpilot](https://www.youtube.com/watch?v=ZEMPQjK1ZPw)
- Add video of goggle maps as input to the model
- Simple transformation to make the map simple (Autoencoder)
- [Valhalla](https://valhalla.github.io/valhalla/api/map-matching/api-reference/) is used to connect gps points from the comma device to create a route map

### Some challenges
- Some challenges with the projection as the earth is sphere and a map is flat
  - To deal with that they adjusted the zoom level to have a constant size
- The model does not learn to handle missing an exit etc.
  - Creates artificial maps with "wrong" turns as part of the route
- Tests, tests, tests
- 


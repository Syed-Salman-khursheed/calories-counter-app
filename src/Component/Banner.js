import * as React from "react";
import { View ,Image, StyleSheet,Dimensions,Text } from "react-native";
const windowWidth = Dimensions.get('window').width;
export default function Banner() {

  return (
      <View style={styles.container}>
          <Image
            style={styles.img}
            source={{uri:'https://thumbs.dreamstime.com/z/calorie-counter-word-concepts-banner-dietary-nutrition-rule-weight-lose-infographics-linear-icons-blue-calorie-counter-183951948.jpg?w=768'}}
          />
      </View>
  );
}
const styles = StyleSheet.create({

  container: {
    paddingHorizontal:15,
    paddingVertical:0,
    width:windowWidth,
    // elevation:30,
  },
  img: {
    borderRadius:7,
    height: 180,
  },
});

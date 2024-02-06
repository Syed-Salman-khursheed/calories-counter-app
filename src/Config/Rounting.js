import * as React from "react";
import 'react-native-gesture-handler';
import Register from "../Screen/Register";
import SignIn from "../Screen/Login";
import Home from "../Screen/Home";
import Registration from "../Screen/Registration";
import Feedback from "../Screen/Feedback";
import DietHistory from "../Screen/Diet";
import CameraScreen from '../Screen/Camera'
import CustomDrawer from "../Component/CustomerDrawer";
import { createStackNavigator } from "@react-navigation/stack";
import { createDrawerNavigator } from '@react-navigation/drawer';
import Splash from "../Screen/Splash";
// import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';

const Stack = createStackNavigator();
const Drawer = createDrawerNavigator();
// const Tab = createBottomTabNavigator();

export default function Rounting() {
  return (
     <MyDrawer/>
      // <MyStack/>
    //  <MyTabs/>
  );
}

function MyStack() {
  return (
    <Stack.Navigator initialRouteName='Splash'>
    <Stack.Screen name="Splash" component={Splash} options={{ headerShown: false }} />
    <Stack.Screen name="Register" component={Register} options={{ headerShown: false }} />
    <Stack.Screen name="Login" component={SignIn} options={{ headerShown: false }}/>
    <Stack.Screen name="Registration" component={Registration} options={{ headerShown: false }} />
    <Stack.Screen name="Home" component={Home} options={{ headerShown: false }} />
      {/* <Stack.Screen name="Camera" component={CustomCamera} options={{ headerShown: false }} /> */}

    </Stack.Navigator>
  );
}

// function MyTabs() {
//   return (
//     <Tab.Navigator >
//       <Tab.Screen name="Home" component={MyDrawer} options={{ headerShown: false }} />
//       <Tab.Screen name="Profile" component={Profile}/>
//       <Tab.Screen name="GoalData" component={GoalData} options={{ headerShown: false }}/>
//     </Tab.Navigator>
//   );
// }




function MyDrawer() {
  return (

    <Drawer.Navigator
      drawerContent={(props) => <CustomDrawer {...props} />}
      screenOptions={{
        drawerActiveBackgroundColor: "#01714b",
        drawerActiveTintColor: "#fff",
        drawerInactiveTintColor: "#01714b",
      }}
    >

      <Drawer.Screen name="Home" component={MyStack}
        options={{
          headerShown: false ,
          headerTitleStyle: {
            color: '#fff',
          },
          headerStyle: {
            backgroundColor: "#01714b",
          },
        }}
        screenOptions={{
          overlayColor: 'transparent',
        }}
      />

      {/* <Drawer.Screen name="History Diet" component={CameraScreen}
        options={{
          headerShown: false ,
          headerTitleStyle: {
            color: '#fff',
          },
          headerStyle: {
            backgroundColor: "#01714b",
          },
        }}
        screenOptions={{
           overlayColor: 'transparent',
        }}
      /> */}
        <Drawer.Screen name="Diet History" component={DietHistory}
        options={{
          headerShown: false ,
          headerTitleStyle: {
            color: '#fff',
          },
          headerStyle: {
            backgroundColor: "#01714b",
          },
        }}
        screenOptions={{
          overlayColor: 'transparent',
        }}
      /> 
      <Drawer.Screen name="Feedback" component={Feedback}
        options={{
          headerShown: false ,
          headerTitleStyle: {
            color: '#fff',
          },
          headerStyle: {
            backgroundColor: "#01714b",
          },
        }}
        screenOptions={{
          overlayColor: 'transparent',
        }}
      /> 
     
  

    </Drawer.Navigator>


  );
}
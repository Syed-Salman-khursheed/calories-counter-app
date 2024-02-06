module.exports = {
    root: true,
    extends: [
      '@react-native-community',
      'plugin:prettier/recommended',
      'plugin:react/recommended',
      'airbnb',
      'airbnb/hooks',
      'prettier',
      'prettier/react',
      'eslint:recommended'
    ],
    parser: 'babel-eslint',
    parserOptions: {ecmaFeatures: {jsx: true}},
    plugins: ['react', 'react-native'],
    env: {
      'react-native/react-native': true,
    },
    rules: {
      'react-native/no-unused-styles': 2,
      'react-native/split-platform-components': 2,
      'react-native/no-inline-styles': 2,
      'react-native/no-color-literals': 2,
      'react-native/no-raw-text': 2,
      'react-native/no-single-element-style-arrays': 2,
    },
  };